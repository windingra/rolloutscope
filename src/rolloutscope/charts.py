"""Tiny dependency-free SVG charts.

The report must be a single offline HTML file: no CDN, no JS chart library, no
matplotlib PNGs. Charts are emitted as inline SVG from pure Python so the file
opens anywhere (and screenshots cleanly).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from html import escape

INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
AXIS = "#d1d5db"
SURFACE = "#ffffff"

CRITICAL = "#dc2626"
WARN = "#d97706"
INFO = "#2563eb"
OK = "#059669"
ACCENT = "#4f46e5"

PALETTE = ["#4f46e5", "#0ea5e9", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6", "#64748b"]


class Svg:
    def __init__(self, width: int, height: int, pad: int = 0):
        self.width = width
        self.height = height
        self.pad = pad
        self._parts: list[str] = []

    def add(self, raw: str) -> Svg:
        self._parts.append(raw)
        return self

    def rect(self, x, y, w, h, fill="none", stroke="none", rx=0, opacity=None, extra=""):
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0):.2f}" height="{max(h, 0):.2f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}"{op} {extra}/>'
        )
        return self

    def line(self, x1, y1, x2, y2, stroke=AXIS, width=1, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width}"{d}{op}/>'
        )
        return self

    def text(self, x, y, content, size=11, fill=INK, anchor="start", weight="400", family="ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial"):
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-family="{family}">{escape(str(content))}</text>'
        )
        return self

    def circle(self, cx, cy, r, fill=ACCENT, stroke="none", opacity=None, width=1):
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width}"{op}/>'
        )
        return self

    def polyline(self, points: Sequence[tuple[float, float]], stroke=ACCENT, width=2, fill="none", dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.add(
            f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round"{d}/>'
        )
        return self

    def polygon(self, points: Sequence[tuple[float, float]], fill=ACCENT, opacity=0.15, stroke="none"):
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.add(f'<polygon points="{pts}" fill="{fill}" opacity="{opacity}" stroke="{stroke}"/>')
        return self

    def render(self) -> str:
        return (
            f'<svg viewBox="0 0 {self.width} {self.height}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial">'
            + "".join(self._parts)
            + "</svg>"
        )


def _nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo]
    span = hi - lo
    raw = span / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * mag
        if step >= raw:
            break
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.5:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:.0f}k"
    if abs(v) >= 10:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def stacked_bars(
    x: Sequence[float],
    series: dict[str, Sequence[float]],
    colors: dict[str, str],
    width: int = 820,
    height: int = 240,
    y_max: float = 1.0,
    line: Sequence[float] | None = None,
    line_label: str = "",
    y_label: str = "",
) -> str:
    """100% stacked bars (fractions) with an optional overlay line (0..1)."""
    left, right, top, bottom = 46, 16, 18, 34
    pw, ph = width - left - right, height - top - bottom
    svg = Svg(width, height)

    for tick in _nice_ticks(0, y_max, 4):
        y = top + ph - (tick / y_max) * ph
        svg.line(left, y, left + pw, y, GRID, 1)
        svg.text(left - 8, y + 3.5, f"{tick:.0%}", 10, MUTED, "end")

    if y_label:
        svg.text(left - 34, top - 6, y_label, 10, MUTED, "start")

    n = len(x)
    if n == 0:
        return svg.render()
    slot = pw / n
    bw = max(3.0, slot * 0.68)

    for i, xv in enumerate(x):
        cx = left + slot * i + slot / 2
        acc = 0.0
        for name, values in series.items():
            v = float(values[i]) if i < len(values) else 0.0
            v = max(0.0, min(1.0, v))
            h = v * ph
            if h <= 0:
                continue
            y = top + ph - (acc + v) * ph
            svg.rect(cx - bw / 2, y, bw, h, fill=colors.get(name, ACCENT))
            acc += v
        if i == 0 or i == n - 1 or (n <= 12) or (i % max(1, n // 10) == 0):
            svg.text(cx, top + ph + 16, f"{int(xv)}", 10, MUTED, "middle")
    svg.text(left + pw / 2, height - 4, "training step", 10, MUTED, "middle")

    if line is not None:
        pts = []
        for i in range(n):
            cx = left + slot * i + slot / 2
            v = float(line[i]) if i < len(line) else 0.0
            pts.append((cx, top + ph - max(0.0, min(1.0, v)) * ph))
        svg.polyline(pts, stroke=INK, width=1.8, dash="4 3")
        if line_label:
            lx, ly = pts[-1]
            svg.text(
                min(lx + 6, width - right - 4),
                max(ly - 10, top + 10),
                line_label,
                10,
                INK,
                "end" if lx > width - 120 else "start",
                "600",
            )

    svg.line(left, top + ph, left + pw, top + ph, AXIS, 1)
    return svg.render()


def line_with_band(
    x: Sequence[float],
    y: Sequence[float],
    lo: Sequence[float] | None = None,
    hi: Sequence[float] | None = None,
    width: int = 820,
    height: int = 240,
    color: str = ACCENT,
    y_label: str = "",
    secondary: Sequence[float] | None = None,
    secondary_label: str = "",
    secondary_color: str = WARN,
    area: bool = True,
) -> str:
    left, right, top, bottom = 52, 46, 18, 34
    pw, ph = width - left - right, height - top - bottom
    svg = Svg(width, height)

    vals = [v for v in y if v is not None]
    vals += [v for v in (lo or []) if v is not None] + [v for v in (hi or []) if v is not None]
    if secondary:
        vals += [v for v in secondary if v is not None]
    if not vals:
        return svg.render()
    vmin, vmax = min(vals), max(vals)
    if vmax <= vmin:
        vmax = vmin + 1
    pad = (vmax - vmin) * 0.08
    vmin, vmax = vmin - pad, vmax + pad

    def sx(i: int) -> float:
        return left + (pw * i / max(1, len(x) - 1))

    def sy(v: float) -> float:
        return top + ph - (v - vmin) / (vmax - vmin) * ph

    for tick in _nice_ticks(vmin, vmax, 4):
        ty = sy(tick)
        svg.line(left, ty, left + pw, ty, GRID, 1)
        svg.text(left - 8, ty + 3.5, _fmt(tick), 10, MUTED, "end")
    if y_label:
        svg.text(left - 44, top - 6, y_label, 10, MUTED, "start")

    if lo is not None and hi is not None:
        pts = [(sx(i), sy(float(hi[i]))) for i in range(len(x))]
        pts += [(sx(i), sy(float(lo[i]))) for i in range(len(x) - 1, -1, -1)]
        svg.polygon(pts, fill=color, opacity=0.10)

    pts = [(sx(i), sy(float(y[i]))) for i in range(len(x))]
    if area:
        base = [(sx(len(x) - 1), top + ph), (sx(0), top + ph)]
        svg.polygon(pts + base, fill=color, opacity=0.06)
    svg.polyline(pts, stroke=color, width=2.2)
    for px, py in pts:
        svg.circle(px, py, 2.4, fill=SURFACE, stroke=color, width=1.6)

    if secondary:
        s_pts = [(sx(i), sy(float(v))) for i, v in enumerate(secondary)]
        svg.polyline(s_pts, stroke=secondary_color, width=1.8, dash="5 3")
        if secondary_label:
            svg.text(width - right + 6, max(top + 10, s_pts[-1][1]), secondary_label, 10, secondary_color, "start", "600")

    n = len(x)
    for i, xv in enumerate(x):
        if n <= 12 or i % max(1, n // 10) == 0 or i == n - 1:
            svg.text(sx(i), top + ph + 16, f"{int(xv)}", 10, MUTED, "middle")
    svg.text(left + pw / 2, height - 4, "training step", 10, MUTED, "middle")
    svg.line(left, top + ph, left + pw, top + ph, AXIS, 1)
    return svg.render()


def scatter(
    xs: Sequence[float],
    ys: Sequence[float],
    highlight: Sequence[bool] | None = None,
    width: int = 820,
    height: int = 260,
    x_label: str = "output tokens",
    y_label: str = "reward",
    highlight_color: str = CRITICAL,
    jitter: float = 0.0,
) -> str:
    left, right, top, bottom = 52, 18, 22, 40
    pw, ph = width - left - right, height - top - bottom
    svg = Svg(width, height)
    if not xs:
        return svg.render()
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin:
        xmax = xmin + 1
    if ymax <= ymin:
        ymax = ymin + 1
    ymin, ymax = ymin - jitter * 2, ymax + jitter * 2

    def sx(v: float) -> float:
        return left + (v - xmin) / (xmax - xmin) * pw

    def sy(v: float) -> float:
        return top + ph - (v - ymin) / (ymax - ymin) * ph

    for tick in _nice_ticks(xmin, xmax, 5):
        x = sx(tick)
        svg.line(x, top, x, top + ph, GRID, 1)
        svg.text(x, top + ph + 15, _fmt(tick), 10, MUTED, "middle")
    for tick in _nice_ticks(ymin, ymax, 4):
        y = sy(tick)
        svg.line(left, y, left + pw, y, GRID, 1)
        svg.text(left - 8, y + 3.5, _fmt(tick), 10, MUTED, "end")

    hl = list(highlight) if highlight is not None else [False] * len(xs)
    for i, (xv, yv) in enumerate(zip(xs, ys)):
        is_hl = bool(hl[i]) if i < len(hl) else False
        # deterministic jitter so dense rows of identical values stay visible
        off = (((i * 2654435761) % 1000) / 1000.0 - 0.5) * 2 * jitter if jitter else 0.0
        svg.circle(
            sx(float(xv)),
            sy(float(yv) + off),
            3.2 if is_hl else 2.0,
            fill=highlight_color if is_hl else "#94a3b8",
            opacity=0.9 if is_hl else 0.35,
            stroke=SURFACE if is_hl else "none",
            width=0.8,
        )
    svg.text(left + pw / 2, height - 4, x_label, 10, MUTED, "middle")
    if y_label:
        svg.text(left - 8, top - 8, y_label, 10, MUTED, "end")
    svg.line(left, top + ph, left + pw, top + ph, AXIS, 1)
    return svg.render()


def ring(value: int, size: int = 132, color: str | None = None) -> str:
    """Health-score ring, 0..100."""
    v = max(0, min(100, int(value)))
    color = color or (OK if v >= 75 else WARN if v >= 50 else CRITICAL)
    svg = Svg(size, size)
    cx = cy = size / 2
    r = size / 2 - 10
    circ = 2 * math.pi * r
    svg.add(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{GRID}" stroke-width="10"/>'
    )
    svg.add(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="10" '
        f'stroke-linecap="round" stroke-dasharray="{circ * v / 100:.2f} {circ:.2f}" '
        f'transform="rotate(-90 {cx} {cy})"/>'
    )
    svg.text(cx, cy + 6, str(v), 30, INK, "middle", "700")
    svg.text(cx, cy + 24, "/ 100", 10, MUTED, "middle")
    return svg.render()


def legend(items: Iterable[tuple[str, str]]) -> str:
    chips = []
    for label, color in items:
        chips.append(
            f'<span class="chip"><i style="background:{color}"></i>{escape(label)}</span>'
        )
    return '<div class="legend">' + "".join(chips) + "</div>"
