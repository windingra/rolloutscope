"""R5 — invalid rollouts: the ones that were never going to teach anything."""

from __future__ import annotations

from collections import Counter

from ...analytics import invalid_reason_series
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

REASON_LABELS = {
    "truncated": "hit the token limit (truncated)",
    "tool_format": "malformed tool call",
    "empty": "empty output",
    "tool_loop": "repeating the same tool call",
    "tool_error": "tool returned an error",
    "unknown": "other",
}

REASON_ACTIONS = {
    "tool_format": (
        "Malformed tool calls usually mean the chat template and the parser disagree "
        "(e.g. the model emits `<tool_call>` in plain text, or the tool schema is not in the prompt). "
        "Dump one failing sample, diff the rendered prompt against the tool schema, and fix the template "
        "before touching the reward."
    ),
    "truncated": (
        "Truncated rollouts waste the whole generation budget and get scored on a cut-off answer. "
        "Raise max_response_length, reduce n, or add a stop condition for the agent loop."
    ),
    "empty": (
        "Empty outputs point at a stop-token / EOS handling bug, or at a prompt that the model refuses. "
        "Check the first few samples of a group."
    ),
    "tool_loop": (
        "Repeated identical tool calls mean the agent loop has no progress check. "
        "Add a repetition guard in the loop (and a small penalty) so the rollout ends instead of burning tokens."
    ),
    "tool_error": (
        "Tool errors are usually environment-side: missing file, bad arguments, timeouts. "
        "Fix the sandbox before interpreting the reward."
    ),
}


class InvalidRolloutRule(Rule):
    """Quantify structurally useless rollouts and say which kind dominates."""

    id = "invalid-rollout"
    title = "Invalid rollouts"
    category = "quality"

    def run(self, ctx: AnalysisContext) -> Finding:
        df = ctx.df
        if df.empty:
            return self.skipped("no traces", self.title, self.id)

        reasons = invalid_reason_series(df)
        bad = reasons != ""
        ratio = float(bad.mean())
        counts = Counter(reasons[bad].tolist())
        wasted_tokens = float(df.loc[bad, "total_tokens"].sum())
        waste = estimate_waste(wasted_tokens, ctx.config)

        if ratio >= 0.15:
            severity = Severity.CRITICAL
        elif ratio >= 0.05:
            severity = Severity.WARN
        elif ratio > 0:
            severity = Severity.INFO
        else:
            severity = Severity.OK

        top = counts.most_common(1)[0][0] if counts else None
        if severity is Severity.OK:
            headline = "No structurally invalid rollouts detected"
        else:
            breakdown = ", ".join(
                f"{REASON_LABELS.get(k, k)} {v} ({v / len(df):.1%})" for k, v in counts.most_common(3)
            )
            headline = (
                f"{pct(ratio)} of rollouts are invalid — {breakdown}; "
                f"≈{fmt_int(wasted_tokens)} tokens wasted"
            )

        actions = []
        if top:
            actions.append(REASON_ACTIONS.get(top, "Inspect the failing samples."))
        if counts.get("tool_format", 0) and counts.get("tool_format", 0) >= 3:
            actions.append(
                "Because this is a template/parser bug, it usually costs one line of config to fix and "
                "immediately raises the effective batch size — fix it before tuning hyper-parameters."
            )

        return Finding(
            rule_id=self.id,
            title=self.title,
            severity=severity,
            headline=headline,
            detail=(
                "An invalid rollout cannot produce a meaningful reward: it was cut off, it never formed a "
                "valid tool call, or it looped. In GRPO these samples still enter the group statistics, so "
                "they drag the group mean and std around and shrink the advantage of the good samples too. "
                "This is the cheapest class of problem to fix and the most under-reported."
            ),
            metrics={
                "invalid_ratio": round(ratio, 4),
                "counts": dict(counts),
                "wasted_tokens": waste["tokens"],
                "wasted_gpu_hours": waste["gpu_hours"],
                "wasted_cost": waste["cost"],
            },
            evidence=_evidence(ctx, df, reasons),
            actions=actions,
            penalty=ctx.config.weight_invalid * min(1.0, ratio / 0.2),
        )


def _evidence(ctx: AnalysisContext, df, reasons) -> list[Evidence]:
    out: list[Evidence] = []
    for reason in ["tool_format", "truncated", "tool_loop", "empty"]:
        subset = df[reasons == reason]
        if subset.empty:
            continue
        row = subset.iloc[0]
        out.append(
            Evidence(
                trace_id=str(row["trace_id"]),
                label=REASON_LABELS.get(reason, reason),
                detail=(
                    f"step {int(row['step'])}: output_tokens={int(row['output_tokens'])}, "
                    f"finish={row['last_finish_reason']}, tool_calls={int(row['num_tool_calls'])}, "
                    f"reward={row['reward']}"
                ),
            )
        )
        if len(out) >= 3:
            break
    return out
