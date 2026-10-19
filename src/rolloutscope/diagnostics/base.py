"""Diagnostic engine primitives: findings, rules, configuration, scoring."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from ..schema import TraceBatch


class Severity(str, Enum):
    CRITICAL = "critical"
    WARN = "warn"
    INFO = "info"
    OK = "ok"

    @property
    def rank(self) -> int:
        return {"critical": 0, "warn": 1, "info": 2, "ok": 3}[self.value]


@dataclass
class Evidence:
    """A concrete rollout the user can open and read."""

    trace_id: str
    label: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Finding:
    """One diagnostic conclusion, always with a number and an action."""

    rule_id: str
    title: str
    severity: Severity
    headline: str
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    #: Points subtracted from the 0-100 rollout health score.
    penalty: float = 0.0
    #: Set when the rule could not run (not enough data, missing column, ...).
    skipped: bool = False
    #: Non-serialized payload for the renderer (e.g. highlighted trace ids).
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "severity": self.severity.value,
            "headline": self.headline,
            "detail": self.detail,
            "metrics": self.metrics,
            "evidence": [e.to_dict() for e in self.evidence],
            "actions": self.actions,
            "penalty": round(self.penalty, 2),
            "skipped": self.skipped,
        }


@dataclass
class Config:
    """All thresholds in one place; overridable via a JSON config file.

    Defaults are deliberately conservative — a rule that fires on noise is worse
    than a rule that stays quiet, because users stop trusting the report.
    """

    #: std(reward) below this counts as "all rollouts identical".
    zero_adv_std: float = 1e-6
    #: reward >= this means "solved". ``None`` = infer from the data.
    pass_threshold: float | None = None
    #: Entropy/NLL ratio versus the first-N-step baseline that counts as collapse.
    entropy_drop_ratio: float = 0.60
    #: Consecutive steps below the ratio before we call it sustained.
    entropy_sustain_steps: int = 2
    #: Steps used to build the baseline.
    baseline_steps: int = 3
    #: Output-length ratio versus baseline that counts as collapse.
    length_drop_ratio: float = 0.60
    #: Within-step length coefficient of variation below this = degenerate.
    length_cv_min: float = 0.05
    #: A "high reward" sample is at or above this within-step quantile.
    hack_reward_quantile: float = 0.90
    #: Flag as short when length < this fraction of the step median.
    hack_length_ratio: float = 0.35
    #: Flag as tool-starved when calls <= this fraction of the step median.
    hack_tool_ratio: float = 0.20
    #: Minimum samples in a step before reward-hacking analysis is trusted.
    hack_min_samples: int = 20
    #: Lift threshold for the "short => high reward" association.
    hack_lift_threshold: float = 1.5
    #: Reward-hack flag rate that maps to a full penalty.
    hack_rate_cap: float = 0.20
    #: Consecutive identical tool calls that count as a loop.
    tool_loop_repeat: int = 3
    #: Health-score weights (must sum to <= 100 in effect; clipped anyway).
    weight_zero_adv: float = 30.0
    weight_invalid: float = 25.0
    weight_reward_hack: float = 20.0
    weight_entropy: float = 15.0
    weight_length: float = 10.0
    #: Seconds per 1k tokens, used to translate waste into GPU-hours.
    tokens_per_second: float = 2500.0
    #: Currency per GPU-hour, only used for the cost estimate.
    cost_per_gpu_hour: float = 2.0
    currency: str = "USD"

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Config:
        cfg = cls()
        for k, v in (raw or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    @classmethod
    def from_file(cls, path: str) -> Config:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisContext:
    batch: TraceBatch
    df: pd.DataFrame
    groups: pd.DataFrame
    steps: pd.DataFrame
    config: Config
    pass_threshold: float = 1.0

    @property
    def has_rewards(self) -> bool:
        return bool(self.df["reward"].notna().any())

    @property
    def has_groups(self) -> bool:
        return self.df["group_key"].nunique() < len(self.df) or "group_id" in self.batch.metadata


class Rule:
    """Base class. Subclasses implement :meth:`run` and return one Finding."""

    id: str = "rule"
    title: str = "Rule"
    category: str = "general"
    description: str = ""

    def run(self, ctx: AnalysisContext) -> Finding:  # pragma: no cover - interface
        raise NotImplementedError

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def skipped(reason: str, title: str | None = None, rule_id: str | None = None) -> Finding:
        return Finding(
            rule_id=rule_id or "rule",
            title=title or "Rule",
            severity=Severity.INFO,
            headline="skipped",
            detail=reason,
            skipped=True,
        )


def estimate_waste(tokens: float, config: Config) -> dict[str, float]:
    """Translate wasted tokens into wall-clock and money."""
    seconds = float(tokens) / max(config.tokens_per_second, 1e-9)
    gpu_hours = seconds / 3600.0
    return {
        "tokens": round(float(tokens), 1),
        "gpu_hours": round(gpu_hours, 3),
        "cost": round(gpu_hours * config.cost_per_gpu_hour, 2),
    }


def pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.1f}%"


def fmt_int(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{int(round(x)):,}"


def sort_findings(findings: Iterable[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (f.severity.rank, -f.penalty, f.rule_id))


def health_score(findings: Iterable[Finding]) -> int:
    penalty = sum(f.penalty for f in findings if not f.skipped)
    return int(max(0, min(100, round(100 - penalty))))
