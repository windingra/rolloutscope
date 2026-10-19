"""Single-file HTML report: no CDN, no JS, opens offline."""

from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from typing import Any

from . import charts
from .analytics import invalid_mask
from .diagnostics.base import Finding, Severity, estimate_waste
from .diagnostics.engine import Report

SEV_COLOR = {
    Severity.CRITICAL: charts.CRITICAL,
    Severity.WARN: charts.WARN,
    Severity.INFO: charts.INFO,
    Severity.OK: charts.OK,
}
SEV_LABEL = {
    Severity.CRITICAL: "CRITICAL",
    Severity.WARN: "WARN",
    Severity.INFO: "INFO",
    Severity.OK: "PASS",
}

CSS = """
:root{
  --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --bg:#f7f8fa; --card:#ffffff;
  --critical:#dc2626; --warn:#d97706; --info:#2563eb; --ok:#059669; --accent:#4f46e5;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,"PingFang SC","Microsoft YaHei",sans-serif;
  -webkit-font-smoothing:antialiased;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:32px 24px 64px}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
.top{display:flex;align-items:center;gap:10px;margin-bottom:18px}
.logo{width:26px;height:26px;border-radius:7px;background:var(--accent);color:#fff;
  display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px}
.brand{font-weight:650;letter-spacing:-.01em}
.brand span{color:var(--muted);font-weight:400}
.hero{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:26px 28px;
  display:flex;gap:28px;align-items:center;justify-content:space-between}
.hero h1{margin:0 0 6px;font-size:24px;letter-spacing:-.02em}
.meta{color:var(--muted);font-size:13px;display:flex;flex-wrap:wrap;gap:8px 14px}
.meta b{color:var(--ink);font-weight:600}
.ring{flex:0 0 132px;text-align:center}
.ring .cap{font-size:11px;color:var(--muted);margin-top:2px;letter-spacing:.04em;text-transform:uppercase}
.grid{display:grid;gap:14px;margin-top:16px}
.kpis{grid-template-columns:repeat(5,1fr)}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:22px;font-weight:650;letter-spacing:-.02em}
.kpi .l{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-top:2px}
.kpi .s{font-size:11px;color:var(--muted);margin-top:6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-top:16px}
.card h2{margin:0 0 2px;font-size:14px;letter-spacing:-.01em}
.card .sub{color:var(--muted);font-size:12px;margin:0 0 12px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
.two .card{margin-top:0}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-top:8px}
.chip{display:inline-flex;align-items:center;gap:6px}
.chip i{width:9px;height:9px;border-radius:2px;display:inline-block}
.findings{margin-top:26px}
.findings h2.sec{font-size:16px;margin:0 0 12px}
.finding{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);
  border-radius:12px;padding:16px 18px;margin-bottom:12px}
.finding.critical{border-left-color:var(--critical)}
.finding.warn{border-left-color:var(--warn)}
.finding.info{border-left-color:var(--info)}
.finding.ok{border-left-color:var(--ok)}
.fhead{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.badge{font-size:10px;font-weight:700;letter-spacing:.06em;padding:2px 7px;border-radius:999px;color:#fff}
.badge.critical{background:var(--critical)} .badge.warn{background:var(--warn)}
.badge.info{background:var(--info)} .badge.ok{background:var(--ok)}
.fhead h3{margin:0;font-size:15px;font-weight:620}
.fhead code{color:var(--muted)}
.headline{margin:8px 0 0;font-size:13.5px}
.metrics{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.m{font-size:11px;background:#f3f4f6;border:1px solid #eceef1;border-radius:6px;padding:2px 7px;color:#374151}
.m b{font-weight:650;color:var(--ink)}
.detail{font-size:12.5px;color:#4b5563;margin:10px 0 0}
.actions{margin-top:12px;font-size:12.5px}
.actions .t{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-bottom:4px}
.actions ul{margin:0;padding-left:18px} .actions li{margin:3px 0}
.evidence{margin-top:12px;border-top:1px dashed var(--line);padding-top:10px}
.evidence .t{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-bottom:6px}
table{border-collapse:collapse;width:100%;font-size:11.5px}
td{padding:4px 8px 4px 0;vertical-align:top;border-bottom:1px solid #f1f2f4}
td.id{white-space:nowrap;color:var(--accent)}
td.lb{white-space:nowrap;color:var(--muted)}
.trace{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px;background:#fcfcfd}
.trace .th{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.pill{font-size:10.5px;border-radius:999px;padding:1px 8px;background:#eef2ff;color:#3730a3}
.pill.bad{background:#fef2f2;color:#991b1b}
.pill.good{background:#ecfdf5;color:#065f46}
.prompt{font-size:12px;color:#374151;background:#fff;border:1px solid var(--line);border-radius:8px;padding:8px 10px;
  white-space:pre-wrap;max-height:120px;overflow:hidden}
.turn{margin-top:8px;padding-left:10px;border-left:2px solid #e5e7eb}
.turn-meta{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.turn-text{font-size:12px;color:#374151;white-space:pre-wrap;margin-top:2px}
.tool{margin-top:5px;font-size:11.5px;background:#fff7ed;border:1px solid #fed7aa;border-radius:7px;padding:5px 8px;
  color:#7c2d12;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre-wrap}
.foot{margin-top:28px;color:var(--muted);font-size:11px;text-align:center;line-height:1.7}
@media (max-width:900px){.kpis{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}
  .hero{flex-direction:column;align-items:flex-start}}
"""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _clip(text: str | None, n: int = 420) -> str:
    if not text:
        return ""
    text = str(text).strip()
    return text if len(text) <= n else text[:n].rstrip() + " …"


def _fmt_num(v: Any, digits: int = 1) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (int,)):
        return f"{v:,}"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:,.{digits}f}"
    return str(v)


def _kpi(value: str, label: str, sub: str = "") -> str:
    sub_html = f'<div class="s">{_esc(sub)}</div>' if sub else ""
    return f'<div class="kpi"><div class="v">{value}</div><div class="l">{_esc(label)}</div>{sub_html}</div>'


def _waste(report: Report) -> dict[str, float]:
    """Tokens that bought nothing: invalid rollouts or members of zero-adv groups."""
    df = report.df
    if df.empty:
        return {"tokens": 0.0, "gpu_hours": 0.0, "cost": 0.0}
    groups = report.groups[["step", "group_key", "zero_adv"]] if len(report.groups) else None
    if groups is not None and len(groups):
        merged = df.merge(groups, on=["step", "group_key"], how="left")
        zero = merged["zero_adv"].fillna(False).astype(bool)
        mask = zero | invalid_mask(merged)
    else:
        mask = invalid_mask(df)
    return estimate_waste(float(df.loc[mask.values, "total_tokens"].sum()), report.config)


def _metrics_chips(metrics: dict[str, Any]) -> str:
    chips = []
    for key, val in metrics.items():
        if key in {"per_step", "drift"} or key.startswith("_"):
            continue
        if val is None or val == "":
            continue
        if isinstance(val, dict):
            continue
        if isinstance(val, float):
            val = f"{val:.4g}" if abs(val) < 1 else f"{val:,.2f}"
        chips.append(f'<span class="m">{_esc(key)} <b>{_esc(val)}</b></span>')
    return '<div class="metrics">' + "".join(chips) + "</div>" if chips else ""


def _finding_html(f: Finding) -> str:
    sev = f.severity.value
    chips = _metrics_chips(f.metrics)
    actions = ""
    if f.actions:
        items = "".join(f"<li>{_esc(a)}</li>" for a in f.actions)
        actions = f'<div class="actions"><div class="t">What to do</div><ul>{items}</ul></div>'
    evidence = ""
    if f.evidence:
        rows = "".join(
            f'<tr><td class="id"><code>{_esc(e.trace_id)}</code></td>'
            f'<td class="lb">{_esc(e.label)}</td><td>{_esc(e.detail)}</td></tr>'
            for e in f.evidence
        )
        evidence = (
            '<div class="evidence"><div class="t">Evidence</div>'
            f"<table>{rows}</table></div>"
        )
    detail = f'<p class="detail">{_esc(f.detail)}</p>' if f.detail and not f.skipped else ""
    return (
        f'<article class="finding {sev}">'
        f'<div class="fhead"><span class="badge {sev}">{SEV_LABEL[f.severity]}</span>'
        f"<h3>{_esc(f.title)}</h3><code>{_esc(f.rule_id)}</code></div>"
        f'<p class="headline">{_esc(f.headline)}</p>'
        f"{chips}{detail}{actions}{evidence}</article>"
    )


def _trace_html(trace, reason: str = "") -> str:
    pills = []
    if trace.reward is not None:
        pills.append(f'<span class="pill good">reward {trace.reward:.2f}</span>')
    pills.append(f'<span class="pill">{trace.output_tokens} tok</span>')
    pills.append(f'<span class="pill">{trace.num_turns} turn(s)</span>')
    if trace.num_tool_calls:
        pills.append(f'<span class="pill">{trace.num_tool_calls} tool call(s)</span>')
    if trace.status != "ok":
        pills.append(f'<span class="pill bad">{_esc(trace.invalid_reason or trace.status)}</span>')
    if reason:
        pills.append(f'<span class="pill">{_esc(reason)}</span>')

    body = []
    if trace.prompt:
        body.append(f'<div class="prompt">{_esc(_clip(trace.prompt, 500))}</div>')
    for step in trace.steps[:4]:
        text = _clip(step.text, 400) or "<span style='color:#9ca3af'>(text not decoded)</span>"
        body.append(
            f'<div class="turn"><div class="turn-meta">turn {step.index + 1} · '
            f"{step.output_tokens} tok"
            + (f" · {_esc(step.finish_reason)}" if step.finish_reason else "")
            + f'</div><div class="turn-text">{text}</div>'
        )
        for call in step.tool_calls[:3]:
            body.append(
                f'<div class="tool">{_esc(call.name)} {_esc(_clip(call.arguments, 220))}'
                + (f"  ->  {_esc(_clip(call.result, 160))}" if call.result else "")
                + (f"  !! {_esc(call.error)}" if call.error else "")
                + "</div>"
            )
        body.append("</div>")
    if len(trace.steps) > 4:
        body.append(f'<div class="turn-meta">… {len(trace.steps) - 4} more turn(s)</div>')

    return (
        f'<div class="trace"><div class="th"><code>{_esc(trace.trace_id)}</code>'
        + "".join(pills)
        + "</div>"
        + "".join(body)
        + "</div>"
    )


def render_html(report: Report, title: str | None = None) -> str:
    summary = report.summary
    waste = _waste(report)
    steps = report.steps
    x = [int(s) for s in steps["step"].tolist()] if len(steps) else []

    # -- chart 1: where the compute went -----------------------------------
    stack = ""
    if x:
        stack = charts.stacked_bars(
            x,
            {
                "saturated": steps["saturated"].fillna(0).tolist(),
                "informative": steps["informative"].fillna(0).tolist(),
                "collapsed": steps["collapsed"].fillna(0).tolist(),
            },
            {
                "saturated": "#f59e0b",
                "informative": "#4f46e5",
                "collapsed": "#ef4444",
            },
            line=steps["informative"].fillna(0).tolist(),
            line_label="ERR",
            y_label="share of groups",
        )
        stack_legend = charts.legend(
            [
                ("informative (0 < pass < 1) — produces gradient", "#4f46e5"),
                ("saturated (all correct) — no gradient", "#f59e0b"),
                ("collapsed (all wrong) — no gradient", "#ef4444"),
            ]
        )
    else:
        stack = '<p class="sub">No per-step rollout data in this input.</p>'
        stack_legend = ""

    # -- chart 2/3: length and entropy -------------------------------------
    length_svg = ""
    if x:
        length_svg = charts.line_with_band(
            x,
            steps["median_length"].fillna(0).tolist(),
            steps["p10_length"].fillna(0).tolist(),
            steps["p90_length"].fillna(0).tolist(),
            width=520,
            height=210,
            y_label="output tokens",
        )
    entropy_finding = report.finding("entropy-collapse")
    entropy_svg = ""
    entropy_note = ""
    if entropy_finding:
        per_step = entropy_finding.metrics.get("per_step") or {}
        if per_step:
            ex = sorted(int(k) for k in per_step)
            ey = [float(per_step[k]) for k in ex]
            entropy_svg = charts.line_with_band(
                ex, ey, width=520, height=210, color=charts.WARN, y_label="entropy / token NLL"
            )
            src = entropy_finding.metrics.get("source")
            entropy_note = (
                "trainer-logged entropy (exact)" if src == "exact" else "mean -log p(token) of sampled tokens (proxy)"
            )
        else:
            entropy_note = "no entropy signal in this input"

    # -- chart 4: reward vs length -----------------------------------------
    scatter_svg = ""
    if len(report.df) and report.df["reward"].notna().any():
        df = report.df.dropna(subset=["reward"])
        if len(df) > 2000:
            df = df.sample(2000, random_state=0)
        hack = report.finding("reward-hacking")
        flagged = set(hack.extra.get("flagged_ids", [])) if hack else set()
        scatter_svg = charts.scatter(
            df["output_tokens"].tolist(),
            df["reward"].tolist(),
            highlight=[str(t) in flagged for t in df["trace_id"]],
            width=820,
            height=250,
            y_label="",
            jitter=0.012,
        )

    # -- drilldown ---------------------------------------------------------
    by_id = {t.trace_id: t for t in report.batch.traces}
    picks: list[tuple[str, str]] = []
    for f in report.findings:
        for e in f.evidence:
            if e.trace_id in by_id and e.trace_id not in {p[0] for p in picks}:
                picks.append((e.trace_id, f.title))
            if len(picks) >= 3:
                break
        if len(picks) >= 3:
            break
    drill = "".join(_trace_html(by_id[tid], reason) for tid, reason in picks) or (
        '<p class="sub">No stored text to drill into. For verl, pass a tokenizer to decode responses.</p>'
    )

    findings_html = "".join(_finding_html(f) for f in report.findings)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run_title = title or report.batch.name
    sev = report.worst

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rolloutscope — {_esc(run_title)}</title>
<style>{CSS}</style></head>
<body><div class="wrap">
  <div class="top"><div class="logo">rs</div>
    <div class="brand">rolloutscope <span>/ rollout diagnostics</span></div></div>

  <section class="hero">
    <div>
      <h1>{_esc(run_title)}</h1>
      <div class="meta">
        <span>framework <b>{_esc(summary['framework'])}</b></span>
        <span>traces <b>{summary['traces']:,}</b></span>
        <span>groups <b>{summary['groups']:,}</b></span>
        <span>steps <b>{summary['steps']}</b></span>
        <span>tokens <b>{summary['total_tokens']:,}</b></span>
        <span>pass threshold <b>{report.pass_threshold:g}</b></span>
        <span>schema <b>v0.1</b></span>
      </div>
      <div class="meta" style="margin-top:8px"><span>worst finding <b style="color:{SEV_COLOR[sev]}">{SEV_LABEL[sev]}</b></span>
        <span>generated {generated}</span></div>
    </div>
    <div class="ring">{charts.ring(report.score)}<div class="cap">rollout health</div></div>
  </section>

  <div class="grid kpis">
    {_kpi(f"{summary['err_recent']:.0%}" if summary['err_recent'] is not None else "n/a", "effective rollout ratio", f"last 3 steps · {summary['err']:.0%} overall" if summary['err'] is not None else "groups that can move the policy")}
    {_kpi(f"{summary['zero_advantage_ratio']:.0%}" if summary['zero_advantage_ratio'] is not None else "n/a", "zero-advantage groups", "all-correct + all-wrong")}
    {_kpi(f"{summary['invalid_ratio']:.1%}" if summary['invalid_ratio'] is not None else "n/a", "invalid rollouts", "truncated / bad tool call / loop")}
    {_kpi(f"{waste['gpu_hours']:.1f} h", "wasted compute", f"≈ {waste['cost']:.2f} {report.config.currency} · {waste['tokens']:,.0f} tok")}
    {_kpi(f"{summary['mean_reward']:.3f}" if summary['mean_reward'] is not None else "n/a", "mean reward", "raw, not normalised")}
  </div>

  <section class="card">
    <h2>Where the compute went</h2>
    <p class="sub">Each bar is one training step, split by group outcome. The dashed line is ERR — the share of groups that actually produce a gradient.</p>
    {stack}
    {stack_legend}
  </section>

  <div class="two">
    <section class="card">
      <h2>Output length</h2>
      <p class="sub">Median per step with the p10–p90 band. A shrinking band means rollouts are converging.</p>
      {length_svg or '<p class="sub">n/a</p>'}
    </section>
    <section class="card">
      <h2>Exploration</h2>
      <p class="sub">{_esc(entropy_note) or "n/a"}</p>
      {entropy_svg or '<p class="sub">n/a</p>'}
    </section>
  </div>

  <section class="card">
    <h2>Reward vs. output length</h2>
    <p class="sub">Red points are the rollouts flagged as suspected reward hacking: top-decile reward with a fraction of the usual work. Reward is jittered vertically so overlapping points stay visible.</p>
    {scatter_svg or '<p class="sub">n/a</p>'}
    {charts.legend([("flagged high-reward / low-effort", charts.CRITICAL), ("other rollouts", "#94a3b8")])}
  </section>

  <section class="findings">
    <h2 class="sec">Diagnostics</h2>
    {findings_html}
  </section>

  <section class="card">
    <h2>Drill-down</h2>
    <p class="sub">Representative rollouts referenced by the findings above.</p>
    {drill}
  </section>

  <div class="foot">
    rolloutscope v0.1 · AgentTrace schema is a superset of the OpenTelemetry GenAI semantic conventions<br>
    every number here is derived from the input files — no training run was modified
  </div>
</div></body></html>"""


def write_html(report: Report, path: str, title: str | None = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(report, title=title))
    return path


def write_json(report: Report, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)
    return path


__all__ = ["render_html", "write_html", "write_json"]
