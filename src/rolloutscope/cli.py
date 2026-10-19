"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from . import __version__
from .adapters import load
from .diagnostics import Config, analyze
from .diagnostics.base import Severity

_SEV_TAG = {
    Severity.CRITICAL: "CRITICAL",
    Severity.WARN: "WARN    ",
    Severity.INFO: "INFO    ",
    Severity.OK: "PASS    ",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rolloutscope",
        description="Diagnose RL rollouts: zero-advantage groups, entropy collapse, reward hacking, wasted compute.",
    )
    p.add_argument("--version", action="version", version=f"rolloutscope {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="generate a synthetic run and write a report (no GPU, no network)")
    d.add_argument("--out", default="demo/out", help="output directory (default: demo/out)")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--steps", type=int, default=16)
    d.add_argument("--prompts", type=int, default=20)
    d.add_argument("--rollouts", type=int, default=8, help="rollouts per prompt (GRPO n)")
    d.add_argument("--open", action="store_true", help="open the report in the browser")

    for name, help_text in (
        ("report", "write an HTML report for a rollout dump"),
        ("scan", "print findings to stdout (CI friendly)"),
    ):
        s = sub.add_parser(name, help=help_text)
        s.add_argument("path", help="directory or file with rollout data")
        s.add_argument("--adapter", default="auto", choices=["auto", "verl", "trl", "generic", "otel", "parquet"])
        s.add_argument("--config", help="JSON file overriding diagnostic thresholds")
        s.add_argument("--pass-threshold", type=float, default=None, help="reward value that counts as solved")
        s.add_argument("--tokenizer", default=None, help="HF tokenizer name/path to decode verl token ids")
        s.add_argument("--max-traces", type=int, default=None)
        if name == "report":
            s.add_argument("--out", default="rolloutscope-report.html")
            s.add_argument("--json", dest="json_out", default=None, help="also write findings as JSON")
            s.add_argument("--title", default=None)
            s.add_argument("--open", action="store_true")
        else:
            s.add_argument("--json", dest="json_out", default=None)
            s.add_argument("--fail-under", type=int, default=None, help="exit 1 when the health score is below this")
            s.add_argument("--quiet", action="store_true", help="only print the one-line summary")

    c = sub.add_parser("convert", help="convert a dump into OTel spans or the native jsonl schema")
    c.add_argument("path")
    c.add_argument("--to", required=True, choices=["otel", "jsonl"])
    c.add_argument("--out", required=True)
    c.add_argument("--adapter", default="auto", choices=["auto", "verl", "trl", "generic", "otel", "parquet"])
    c.add_argument("--tokenizer", default=None)

    return p


def _load(args) -> object:
    kwargs = {}
    if getattr(args, "tokenizer", None):
        kwargs["tokenizer"] = args.tokenizer
    if getattr(args, "max_traces", None):
        kwargs["max_traces"] = args.max_traces
    return load(args.path, adapter=args.adapter, **kwargs)


def _config(args) -> Config:
    cfg = Config.from_file(args.config) if getattr(args, "config", None) else Config()
    if getattr(args, "pass_threshold", None) is not None:
        cfg.pass_threshold = args.pass_threshold
    return cfg


def _print_report(report, quiet: bool = False) -> None:
    s = report.summary
    print(
        f"rolloutscope {__version__}  ·  {s['framework']}  ·  "
        f"{s['traces']:,} traces / {s['groups']:,} groups / {s['steps']} steps  ·  "
        f"{s['total_tokens']:,} tokens"
    )
    err = "n/a" if s["err"] is None else f"{s['err']:.0%}"
    za = "n/a" if s["zero_advantage_ratio"] is None else f"{s['zero_advantage_ratio']:.0%}"
    inv = "n/a" if s["invalid_ratio"] is None else f"{s['invalid_ratio']:.1%}"
    print(f"health {report.score}/100   ERR {err}   zero-advantage {za}   invalid {inv}")
    if quiet:
        return
    print("-" * 96)
    for f in report.findings:
        if f.skipped:
            continue
        print(f"[{_SEV_TAG[f.severity]}] {f.rule_id:<18} {f.headline}")
    if report.errors:
        print("-" * 96)
        for e in report.errors:
            print(f"rule error: {e}")


def _open(path: str) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
    except Exception:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "demo":
        from .demo import DemoConfig, write_demo

        cfg = DemoConfig(steps=args.steps, prompts=args.prompts, rollouts=args.rollouts, seed=args.seed)
        paths = write_demo(args.out, cfg)
        print(f"demo written to {os.path.abspath(args.out)}")
        for key, value in paths.items():
            print(f"  {key:<12} {value}")
        print()
        print("next:")
        print(f"  rolloutscope report {paths['verl_dump']}     # raw verl-shaped dump (token ids only)")
        print(f"  rolloutscope report {paths['trl']}           # TRL trainer_state.json + completions")
        print(f"  rolloutscope report {paths['spans']} --adapter otel")
        if args.open:
            _open(os.path.abspath(paths["report"]))
        return 0

    try:
        batch = _load(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = analyze(batch, _config(args))

    if args.command == "scan":
        _print_report(report, quiet=args.quiet)
        if args.json_out:
            from .report import write_json

            write_json(report, args.json_out)
            print(f"json -> {args.json_out}")
        if args.fail_under is not None and report.score < args.fail_under:
            print(f"health {report.score} < --fail-under {args.fail_under}", file=sys.stderr)
            return 1
        return 0

    if args.command == "report":
        from .report import write_html, write_json

        path = write_html(report, args.out, title=args.title)
        _print_report(report, quiet=False)
        print(f"report -> {os.path.abspath(path)}")
        if args.json_out:
            write_json(report, args.json_out)
            print(f"json   -> {os.path.abspath(args.json_out)}")
        if args.open:
            _open(os.path.abspath(path))
        return 0

    if args.command == "convert":
        if args.to == "otel":
            from .otel import to_otlp

            payload = to_otlp(batch)
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            n = sum(len(ss["spans"]) for rs in payload["resourceSpans"] for ss in rs["scopeSpans"])
            print(f"wrote {n:,} spans -> {args.out}")
        else:
            batch.to_jsonl(args.out)
            print(f"wrote {len(batch):,} traces -> {args.out}")
        return 0

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
