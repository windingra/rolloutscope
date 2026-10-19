"""Rule behaviour on hand-built batches with known pathologies."""

from __future__ import annotations

import pytest

from rolloutscope.diagnostics import Config, analyze
from rolloutscope.diagnostics.base import Severity


def test_healthy_run_is_clean(healthy_batch):
    report = analyze(healthy_batch)
    zero = report.finding("zero-advantage")
    assert zero is not None
    assert zero.metrics["zero_advantage_ratio"] == pytest.approx(0.0)
    assert zero.severity is Severity.OK
    assert report.finding("invalid-rollout").metrics["invalid_ratio"] == pytest.approx(0.0)
    assert report.score >= 90


def test_zero_advantage_splits_saturated_and_collapsed(broken_batch):
    report = analyze(broken_batch)
    zero = report.finding("zero-advantage")
    assert zero.metrics["zero_advantage_ratio"] == pytest.approx(0.5, abs=0.01)
    assert zero.metrics["saturated_ratio"] == pytest.approx(0.25, abs=0.01)
    assert zero.metrics["collapsed_ratio"] == pytest.approx(0.25, abs=0.01)
    assert zero.metrics["wasted_tokens"] > 0
    assert zero.severity is Severity.CRITICAL
    # both failure modes must be represented in the evidence, not just the total
    labels = " ".join(e.label for e in zero.evidence)
    assert "saturated" in labels and "collapsed" in labels


def test_effective_rollout_ratio(broken_batch):
    report = analyze(broken_batch)
    finding = report.finding("group-difficulty")
    assert finding.metrics["err"] == pytest.approx(0.5, abs=0.01)
    assert finding.metrics["gradient_density"] is not None


def test_invalid_rollouts_are_classified(broken_batch):
    report = analyze(broken_batch)
    finding = report.finding("invalid-rollout")
    counts = finding.metrics["counts"]
    assert counts.get("tool_format", 0) == 4
    assert counts.get("truncated", 0) == 4
    assert finding.metrics["invalid_ratio"] == pytest.approx(8 / 136, abs=0.01)
    assert finding.severity in (Severity.WARN, Severity.CRITICAL)


def test_reward_hacking_flags_the_short_high_reward_rollout(broken_batch):
    report = analyze(broken_batch)
    finding = report.finding("reward-hacking")
    assert finding.metrics["flagged"] >= 1
    ids = {e.trace_id for e in finding.evidence}
    assert any(t.startswith("s0-mix-a-n0") for t in ids)


def test_length_collapse_needs_a_drop(broken_batch):
    report = analyze(broken_batch)
    finding = report.finding("length-collapse")
    # lengths are constant in this fixture, so it must not claim a collapse
    assert finding.metrics["min_ratio"] == pytest.approx(1.0, abs=0.01)


def test_entropy_proxy_decline_is_reported(broken_batch):
    report = analyze(broken_batch)
    finding = report.finding("entropy-collapse")
    assert finding.metrics["source"] == "proxy"
    assert finding.metrics["min_ratio"] == pytest.approx(0.5, abs=0.05)


def test_thresholds_are_configurable(healthy_batch):
    strict = Config(zero_adv_std=1e9)  # everything looks like zero variance
    report = analyze(healthy_batch, strict)
    assert report.finding("zero-advantage").metrics["zero_advantage_ratio"] == 1.0


def test_metrics_only_input_still_scores():
    from rolloutscope.schema import TraceBatch

    batch = TraceBatch(
        traces=[],
        name="trl-state",
        framework="trl",
        metrics={
            "entropy": {0: 3.0, 1: 2.9, 2: 1.7, 3: 1.2},
            "completions/mean_length": {0: 400, 1: 380, 2: 200, 3: 120},
            "frac_reward_zero_std": {0: 0.2, 1: 0.3, 2: 0.5, 3: 0.6},
        },
    )
    report = analyze(batch)
    zero = report.finding("zero-advantage")
    assert zero.metrics["source"] == "trainer_metric"
    assert zero.metrics["zero_advantage_ratio"] == pytest.approx(0.4, abs=0.01)
    assert report.finding("entropy-collapse").severity is Severity.CRITICAL
    assert report.finding("length-collapse").metrics["min_ratio"] < 0.5


def test_report_json_is_serializable(broken_batch):
    import json

    report = analyze(broken_batch)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["summary"]["traces"] == 136
    assert len(payload["findings"]) == 6
    assert all("headline" in f for f in payload["findings"])
