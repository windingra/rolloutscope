"""R2 — entropy collapse: the policy stops exploring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import AnalysisContext, Evidence, Finding, Rule, Severity, pct


class EntropyCollapseRule(Rule):
    """Track per-step entropy and flag a sustained drop against an early baseline.

    Two data sources, in priority order:

    1. ``entropy`` scalar logged by the trainer (TRL logs it, verl logs
       ``actor/entropy``) — labelled ``exact``;
    2. mean ``-log p(token)`` over sampled tokens — labelled ``proxy``. It is the
       NLL of the sampled tokens, i.e. an upper bound on the policy entropy, so
       it tracks collapse but should not be quoted as entropy itself.
    """

    id = "entropy-collapse"
    title = "Entropy collapse"
    category = "exploration"

    def run(self, ctx: AnalysisContext) -> Finding:
        series, source = _entropy_series(ctx)
        if series is None or len(series) < 3:
            return self.skipped(
                "need at least 3 training steps with entropy or token logprobs",
                self.title,
                self.id,
            )

        base_n = min(ctx.config.baseline_steps, max(1, len(series) // 3))
        baseline = float(np.median(series.iloc[:base_n].to_numpy()))
        if baseline <= 0:
            return self.skipped("entropy baseline is zero", self.title, self.id)

        ratio = series / baseline
        breach = ratio < ctx.config.entropy_drop_ratio
        first_breach = int(ratio[breach].index.min()) if breach.any() else None
        sustained = _max_consecutive(breach) >= ctx.config.entropy_sustain_steps
        min_ratio = float(ratio.min())
        depth = max(0.0, 1.0 - min_ratio) if breach.any() else 0.0

        cv = _within_step_cv(ctx)
        homogenized = cv is not None and cv < ctx.config.length_cv_min

        if sustained and min_ratio < 0.5:
            severity = Severity.CRITICAL
        elif sustained:
            severity = Severity.WARN
        elif breach.any():
            severity = Severity.INFO
        else:
            severity = Severity.OK

        label = "entropy" if source == "exact" else "token NLL (proxy)"
        if severity is Severity.OK:
            headline = f"{label} stayed within {pct(ctx.config.entropy_drop_ratio)} of baseline — exploration looks fine"
        else:
            headline = (
                f"{label} dropped to {min_ratio:.0%} of the early baseline"
                + (f", first below threshold at step {first_breach}" if first_breach is not None else "")
                + (" (sustained)" if sustained else " (single step)")
            )

        actions = []
        if severity is not Severity.OK:
            actions.append(
                "Inspect the rollouts around step "
                f"{first_breach if first_breach is not None else 'the drop'}: are completions collapsing onto "
                "one template/answer? Check `output_tokens` CV and the top repeated completions."
            )
            actions.append(
                "Typical fixes: raise the KL coefficient or the sampling temperature, lower the learning rate, "
                "add an entropy bonus, or mix in harder prompts so the policy cannot saturate."
            )
        if homogenized:
            actions.append(
                "Within-step entropy variance is near zero — outputs are nearly identical across rollouts. "
                "This usually precedes full mode collapse."
            )

        return Finding(
            rule_id=self.id,
            title=self.title,
            severity=severity,
            headline=headline,
            detail=(
                "Entropy collapse means the policy has become near-deterministic: all n rollouts in a group "
                "start to look the same, so group-relative advantages shrink toward zero even when the rewards "
                "are not identical. It is the usual precursor of 'reward plateaus while loss keeps dropping'."
            ),
            metrics={
                "source": source,
                "baseline": round(baseline, 4),
                "min_ratio": round(min_ratio, 4),
                "first_breach_step": first_breach,
                "sustained": bool(sustained),
                "within_step_cv": None if cv is None else round(cv, 4),
                "per_step": {int(k): round(float(v), 4) for k, v in series.items()},
            },
            evidence=_evidence(ctx, series, first_breach, source),
            actions=actions,
            penalty=ctx.config.weight_entropy * depth * (1.0 if sustained else 0.5),
        )


def _entropy_series(ctx: AnalysisContext) -> tuple[pd.Series | None, str]:
    logged = ctx.batch.metrics.get("entropy") or ctx.batch.metrics.get("actor/entropy")
    if logged:
        series = pd.Series({int(k): float(v) for k, v in logged.items()}).sort_index()
        series = series[series.notna()]
        if len(series) >= 3:
            return series, "exact"
    if "mean_nll" in ctx.df and ctx.df["mean_nll"].notna().any():
        series = ctx.df.groupby("step")["mean_nll"].median().dropna().sort_index()
        if len(series) >= 3:
            return series, "proxy"
    return None, "none"


def _within_step_cv(ctx: AnalysisContext) -> float | None:
    if "mean_nll" not in ctx.df or not ctx.df["mean_nll"].notna().any():
        return None
    per_step = ctx.df.dropna(subset=["mean_nll"]).groupby("step")["mean_nll"]
    cvs = per_step.std() / per_step.mean().replace(0, np.nan)
    cvs = cvs.dropna()
    return float(cvs.median()) if len(cvs) else None


def _max_consecutive(mask: pd.Series) -> int:
    best = run = 0
    for flag in mask.tolist():
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def _evidence(
    ctx: AnalysisContext, series: pd.Series, first_breach: int | None, source: str
) -> list[Evidence]:
    if first_breach is None or source != "proxy":
        return []
    out: list[Evidence] = []
    for step, label in ((int(series.index[0]), "baseline step"), (first_breach, "collapse step")):
        subset = ctx.df[ctx.df["step"] == step].dropna(subset=["mean_nll"])
        if subset.empty:
            continue
        # Baseline: the rollouts closest to the step median (typical behaviour).
        # Collapse: the lowest, i.e. the most over-confident rollout.
        median = float(subset["mean_nll"].median())
        if label.startswith("baseline"):
            subset = subset.assign(_d=(subset["mean_nll"] - median).abs()).sort_values("_d")
        else:
            subset = subset.sort_values("mean_nll")
        row = subset.iloc[0]
        out.append(
            Evidence(
                trace_id=str(row["trace_id"]),
                label=label,
                detail=(
                    f"step {step}: token NLL={row['mean_nll']:.3f}, "
                    f"output_tokens={int(row['output_tokens'])}, reward={row['reward']}"
                ),
            )
        )
    return out
