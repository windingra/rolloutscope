"""R1 — zero-advantage groups: the rollouts GRPO silently throws away."""

from __future__ import annotations

import pandas as pd

from ...analytics import advantage_zero_ratio
from ..base import (
    AnalysisContext,
    Evidence,
    Finding,
    Rule,
    Severity,
    estimate_waste,
    fmt_int,
    pct,
)


class ZeroAdvantageRule(Rule):
    """Detect groups where every rollout got the same reward.

    For a group of ``n`` rollouts with rewards ``r_1..r_n``, GRPO uses

        A_i = (r_i - mean(r)) / (std(r) + eps)

    When ``std(r) == 0`` every ``A_i`` is exactly zero, so the group contributes
    nothing to the gradient — but its forward passes, reward-model calls and
    tool executions have already been paid for. Splitting the wasted groups into
    *saturated* (all correct) and *collapsed* (all wrong) matters because the
    fix is opposite: drop easy prompts vs. add hints / lower difficulty.
    """

    id = "zero-advantage"
    title = "Zero-advantage groups"
    category = "gradient"

    def run(self, ctx: AnalysisContext) -> Finding:
        if not ctx.has_rewards:
            logged = ctx.batch.metrics.get("frac_reward_zero_std") or {}
            if logged:
                return _from_logged_metric(ctx, logged)
            return self.skipped("no reward column found in the traces", self.title, self.id)

        groups = ctx.groups
        if groups.empty or len(groups) < 2:
            logged = ctx.batch.metrics.get("frac_reward_zero_std") or {}
            if logged:
                return _from_logged_metric(ctx, logged)
            return self.skipped(
                "fewer than 2 groups; GRPO grouping needs n>1 rollouts per prompt", self.title, self.id
            )

        ratio = advantage_zero_ratio(groups)
        sat = float(groups["saturated"].mean())
        col = float(groups["collapsed"].mean())
        wasted_tokens = float(groups.loc[groups["zero_adv"], "tokens"].sum())
        waste = estimate_waste(wasted_tokens, ctx.config)

        per_step = groups.groupby("step")["zero_adv"].mean()
        worst_step = int(per_step.idxmax()) if len(per_step) else 0
        worst_ratio = float(per_step.max()) if len(per_step) else 0.0
        first_step = int(per_step[per_step >= 0.30].index.min()) if (per_step >= 0.30).any() else None

        if ratio >= 0.40:
            severity = Severity.CRITICAL
        elif ratio >= 0.15:
            severity = Severity.WARN
        elif ratio > 0.02:
            severity = Severity.INFO
        else:
            severity = Severity.OK

        headline = (
            f"{pct(ratio)} of groups had zero advantage "
            f"({pct(sat)} all-correct, {pct(col)} all-wrong) — "
            f"{fmt_int(wasted_tokens)} tokens, ≈{waste['gpu_hours']:.2f} GPU-h wasted"
            if severity is not Severity.OK
            else f"Only {pct(ratio)} of groups had zero advantage — gradient signal is healthy"
        )

        actions = []
        if sat > 0.05:
            actions.append(
                f"{pct(sat)} of groups are saturated (all rollouts solved). Down-sample or retire these "
                "prompts, or raise difficulty — they consume n× compute for zero gradient."
            )
        if col > 0.05:
            actions.append(
                f"{pct(col)} of groups are collapsed (no rollout solved). Lower difficulty, inject a hint "
                "or a worked example into the prompt, or use a dynamic baseline so all-wrong groups still "
                "produce signal."
            )
        if severity is Severity.OK:
            actions.append("No action needed; keep monitoring this ratio across steps.")

        evidence = _pick_evidence(ctx, groups)

        return Finding(
            rule_id=self.id,
            title=self.title,
            severity=severity,
            headline=headline,
            detail=(
                "GRPO normalises rewards inside each group: A_i = (r_i - mean) / (std + eps). "
                "If all n rollouts in a group receive the same reward, std = 0 and every advantage "
                "collapses to 0 — the group is pure compute waste, not a learning signal. "
                "The two failure modes look identical in a scalar metric like TRL's "
                "`frac_reward_zero_std` but need opposite fixes, which is why they are split here."
            ),
            metrics={
                "zero_advantage_ratio": round(ratio, 4),
                "saturated_ratio": round(sat, 4),
                "collapsed_ratio": round(col, 4),
                "groups": len(groups),
                "wasted_tokens": waste["tokens"],
                "wasted_gpu_hours": waste["gpu_hours"],
                "wasted_cost": waste["cost"],
                "worst_step": worst_step,
                "worst_step_ratio": round(worst_ratio, 4),
                "first_breach_step": first_step,
            },
            evidence=evidence,
            actions=actions,
            penalty=ctx.config.weight_zero_adv * min(1.0, ratio / 0.5),
        )


def _from_logged_metric(ctx: AnalysisContext, logged: dict) -> Finding:
    """Fallback for a metrics-only run (e.g. TRL ``trainer_state.json``).

    TRL already logs ``frac_reward_zero_std``; we can score it, but we cannot
    split saturated vs. collapsed without per-sample rewards. Say so explicitly
    instead of guessing.
    """
    series = pd.Series({int(k): float(v) for k, v in logged.items()}).sort_index()
    series = series[series.notna()]
    if series.empty:
        return Rule.skipped("no usable frac_reward_zero_std values", "Zero-advantage groups", "zero-advantage")
    ratio = float(series.mean())
    worst_step = int(series.idxmax())
    worst = float(series.max())
    severity = (
        Severity.CRITICAL if ratio >= 0.40 else Severity.WARN if ratio >= 0.15 else Severity.INFO if ratio > 0.02 else Severity.OK
    )
    return Finding(
        rule_id="zero-advantage",
        title="Zero-advantage groups",
        severity=severity,
        headline=(
            f"Trainer reported frac_reward_zero_std={ratio:.1%} on average "
            f"(worst {worst:.1%} at step {worst_step}) — aggregate only, no per-sample split"
        ),
        detail=(
            "Read from the trainer's own scalar metric. Without stored completions we cannot tell "
            "all-correct groups from all-wrong ones, which is exactly the split that decides the fix. "
            "Export completions (TRL: `log_completions=True`) to unlock per-group attribution."
        ),
        metrics={
            "zero_advantage_ratio": round(ratio, 4),
            "worst_step": worst_step,
            "worst_step_ratio": round(worst, 4),
            "source": "trainer_metric",
            "per_step": {int(k): round(float(v), 4) for k, v in series.items()},
        },
        actions=[
            "Re-run with completion logging (or dump the rollout batch) to get the saturated/collapsed split.",
        ],
        penalty=ctx.config.weight_zero_adv * min(1.0, ratio / 0.5),
    )


def _pick_evidence(ctx: AnalysisContext, groups: pd.DataFrame) -> list[Evidence]:
    out: list[Evidence] = []
    for label, mask in (
        ("saturated group (all correct)", groups["saturated"]),
        ("collapsed group (all wrong)", groups["collapsed"]),
    ):
        subset = groups[mask]
        if subset.empty:
            continue
        row = subset.sort_values("tokens", ascending=False).iloc[0]
        trace = ctx.df[
            (ctx.df["group_key"] == row["group_key"]) & (ctx.df["step"] == row["step"])
        ]
        if trace.empty:
            continue
        first = trace.iloc[0]
        rewards = ", ".join(f"{r:.2f}" for r in trace["reward"].dropna().tolist()[:6])
        out.append(
            Evidence(
                trace_id=str(first["trace_id"]),
                label=label,
                detail=(
                    f"step {int(row['step'])}, group {row['group_key']}: "
                    f"n={int(row['n'])}, rewards=[{rewards}], "
                    f"tokens={int(row['tokens']):,}"
                ),
            )
        )
    return out
