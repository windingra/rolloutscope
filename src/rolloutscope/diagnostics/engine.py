"""Diagnostic engine: run every rule over a batch and score the run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..analytics import (
    advantage_zero_ratio,
    group_frame,
    infer_pass_threshold,
    invalid_mask,
    step_frame,
    trace_frame,
)
from ..schema import TraceBatch
from .base import (
    AnalysisContext,
    Config,
    Finding,
    Severity,
    health_score,
    sort_findings,
)
from .rules import ALL_RULES


@dataclass
class Report:
    """The full result of one analysis: findings + the frames used for charts."""

    batch: TraceBatch
    findings: list[Finding]
    score: int
    config: Config
    pass_threshold: float
    df: pd.DataFrame
    groups: pd.DataFrame
    steps: pd.DataFrame
    rules_run: int = 0
    errors: list[str] = field(default_factory=list)

    # -- convenience -------------------------------------------------------
    @property
    def summary(self) -> dict[str, Any]:
        err = float(self.groups["informative"].mean()) if len(self.groups) else None
        err_recent = None
        if len(self.groups):
            per_step = self.groups.groupby("step")["informative"].mean()
            if len(per_step):
                err_recent = float(per_step.tail(3).mean())
        return {
            "name": self.batch.name,
            "framework": self.batch.framework,
            "traces": len(self.batch),
            "groups": len(self.groups),
            "steps": int(self.df["step"].nunique()) if len(self.df) else 0,
            "score": self.score,
            "pass_threshold": self.pass_threshold,
            "err": None if err is None else round(err, 4),
            "err_recent": None if err_recent is None else round(err_recent, 4),
            "zero_advantage_ratio": round(advantage_zero_ratio(self.groups), 4)
            if len(self.groups)
            else None,
            "invalid_ratio": round(float(invalid_mask(self.df).mean()), 4) if len(self.df) else None,
            "total_tokens": int(self.df["total_tokens"].sum()) if len(self.df) else 0,
            "mean_reward": round(float(self.df["reward"].mean()), 4)
            if len(self.df) and self.df["reward"].notna().any()
            else None,
        }

    @property
    def worst(self) -> Severity:
        return min((f.severity for f in self.findings), key=lambda s: s.rank, default=Severity.OK)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "score": self.score,
            "pass_threshold": self.pass_threshold,
            "findings": [f.to_dict() for f in self.findings],
            "config": self.config.to_dict(),
            "errors": self.errors,
        }

    def finding(self, rule_id: str) -> Finding | None:
        return next((f for f in self.findings if f.rule_id == rule_id), None)


def analyze(batch: TraceBatch, config: Config | None = None) -> Report:
    """Run all rules over ``batch`` and return a scored report."""
    cfg = config or Config()
    df = trace_frame(batch)
    threshold = cfg.pass_threshold
    if threshold is None:
        threshold = infer_pass_threshold(df["reward"]) if len(df) else 1.0
    groups = group_frame(df, threshold, cfg.zero_adv_std)
    steps = step_frame(df, groups)
    ctx = AnalysisContext(
        batch=batch, df=df, groups=groups, steps=steps, config=cfg, pass_threshold=threshold
    )

    findings: list[Finding] = []
    errors: list[str] = []
    for rule in ALL_RULES:
        try:
            findings.append(rule.run(ctx))
        except Exception as exc:  # a broken rule must never kill the report
            errors.append(f"{rule.id}: {type(exc).__name__}: {exc}")
            findings.append(
                Finding(
                    rule_id=rule.id,
                    title=rule.title,
                    severity=Severity.INFO,
                    headline="rule failed to run",
                    detail=f"{type(exc).__name__}: {exc}",
                    skipped=True,
                )
            )

    ordered = sort_findings(findings)
    return Report(
        batch=batch,
        findings=ordered,
        score=health_score(ordered),
        config=cfg,
        pass_threshold=threshold,
        df=df,
        groups=groups,
        steps=steps,
        rules_run=len(ALL_RULES),
        errors=errors,
    )


__all__ = ["Report", "analyze"]
