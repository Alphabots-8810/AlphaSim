"""A very small SVG chart builder for the report.

The report's figures are generated, not pasted: every number in them comes
from the solver at build time. Output is plain inline SVG whose colours are CSS
variables, so figures follow the page's light / dark theme and stay sharp at
any zoom. No matplotlib in the page, no raster images except heat maps.
"""
from __future__ import annotations

import html
import itertools
import math
from typing import Callable, Iterable, Optional, Sequence

import numpy as np


_CLIP_IDS = itertools.count(1)


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


class Fig:
    """One plot area with linear axes. Coordinates passed to the drawing
    methods are data coordinates; `px` variants take pixels."""

    def __init__(self, width=760, height=380, left=58, right=20, top=20, bottom=46,
                 xlim=(0.0, 1.0), ylim=(0.0, 1.0), ylog=False):
        self.w, self.h = width, height
        self.l, self.r, self.t, self.b = left, right, top, bottom
        self.xlim, self.ylim, self.ylog = xlim, ylim, ylog
        self.parts: list[str] = []
        self.clip = f"plot{next(_CLIP_IDS)}"      # curves are clipped to the plot area

    # -- coordinate transforms --
    def X(self, x):
        x0, x1 = self.xlim
        return self.l + (np.asarray(x, dtype=float) - x0) / (x1 - x0) * (self.w - self.l - self.r)

    def Y(self, y):
        y0, y1 = self.ylim
        y = np.asarray(y, dtype=float)
        if self.ylog:
            frac = (np.log10(np.maximum(y, 1e-300)) - math.log10(y0)) / (math.log10(y1) - math.log10(y0))
        else:
            frac = (y - y0) / (y1 - y0)
        return self.h - self.b - frac * (self.h - self.t - self.b)

    # -- furniture --
    def axes(self, xticks: Sequence[float], yticks: Sequence[float],
             xfmt: Callable[[float], str] = _fmt, yfmt: Callable[[float], str] = _fmt,
             xlabel: str = "", ylabel: str = "", grid_x: bool = False) -> "Fig":
        x_lo, x_hi = self.l, self.w - self.r
        y_lo, y_hi = self.h - self.b, self.t
        for yt in yticks:
            py = float(self.Y(yt))
            self.parts.append(f'<line class="grid" x1="{x_lo}" x2="{x_hi}" y1="{py:.1f}" y2="{py:.1f}"/>')
            self.parts.append(f'<text class="tick" x="{x_lo - 8}" y="{py + 4:.1f}" text-anchor="end">'
                              f'{html.escape(yfmt(yt))}</text>')
        for xt in xticks:
            px = float(self.X(xt))
            if grid_x:
                self.parts.append(f'<line class="grid" x1="{px:.1f}" x2="{px:.1f}" y1="{y_lo}" y2="{y_hi}"/>')
            self.parts.append(f'<line class="axis" x1="{px:.1f}" x2="{px:.1f}" y1="{y_lo}" y2="{y_lo + 5}"/>')
            self.parts.append(f'<text class="tick" x="{px:.1f}" y="{y_lo + 19}" text-anchor="middle">'
                              f'{html.escape(xfmt(xt))}</text>')
        self.parts.append(f'<line class="axis" x1="{x_lo}" x2="{x_hi}" y1="{y_lo}" y2="{y_lo}"/>')
        if xlabel:
            self.parts.append(f'<text class="axlabel" x="{(x_lo + x_hi) / 2:.1f}" y="{self.h - 6}" '
                              f'text-anchor="middle">{html.escape(xlabel)}</text>')
        if ylabel:
            self.parts.append(f'<text class="axlabel" x="{x_lo}" y="{y_hi - 7}" text-anchor="start">'
                              f'{html.escape(ylabel)}</text>')
        return self

    # -- marks --
    def _runs(self, xs, ys) -> Iterable[str]:
        """Polyline point strings, split wherever the data has a gap (NaN)."""
        px, py = self.X(xs), self.Y(ys)
        run: list[str] = []
        for a, b in zip(np.atleast_1d(px), np.atleast_1d(py)):
            if np.isfinite(a) and np.isfinite(b):
                run.append(f"{a:.1f},{b:.1f}")
            elif run:
                yield " ".join(run)
                run = []
        if run:
            yield " ".join(run)

    def line(self, xs, ys, color: str, width: float = 2.0, dash: Optional[str] = None,
             opacity: float = 1.0) -> "Fig":
        extra = (f' stroke-dasharray="{dash}"' if dash else "") + (
            f' opacity="{opacity}"' if opacity != 1 else "")
        for pts in self._runs(xs, ys):
            self.parts.append(f'<polyline clip-path="url(#{self.clip})" points="{pts}" fill="none" stroke="{color}" '
                              f'stroke-width="{width}" stroke-linejoin="round" '
                              f'stroke-linecap="round"{extra}/>')
        return self

    def area(self, xs, y_low, y_high, color: str, opacity: float = 0.2) -> "Fig":
        xs = np.asarray(xs, dtype=float)
        lo = np.broadcast_to(np.asarray(y_low, dtype=float), xs.shape)
        hi = np.broadcast_to(np.asarray(y_high, dtype=float), xs.shape)
        ok = np.isfinite(lo) & np.isfinite(hi)
        if ok.sum() < 2:
            return self
        top = [f"{a:.1f},{b:.1f}" for a, b in zip(self.X(xs[ok]), self.Y(hi[ok]))]
        bot = [f"{a:.1f},{b:.1f}" for a, b in zip(self.X(xs[ok])[::-1], self.Y(lo[ok])[::-1])]
        self.parts.append(f'<polygon points="{" ".join(top + bot)}" fill="{color}" opacity="{opacity}"/>')
        return self

    def dots(self, xs, ys, color: str, r: float = 3.5, stroke: str = "var(--ink-1)",
             opacity: float = 1.0, yerr=None) -> "Fig":
        xs, ys = np.atleast_1d(xs), np.atleast_1d(ys)
        errs = np.atleast_1d(yerr) if yerr is not None else [None] * len(xs)
        for x, y, e in zip(xs, ys, errs):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            px, py = float(self.X(x)), float(self.Y(y))
            if e is not None and e > 0:
                y_a, y_b = float(self.Y(y - e)), float(self.Y(y + e))
                self.parts.append(f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{y_a:.1f}" y2="{y_b:.1f}" '
                                  f'stroke="{color}" stroke-width="1.5"/>')
            self.parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r}" fill="{color}" '
                              f'stroke="{stroke}" stroke-width="1.2" opacity="{opacity}"/>')
        return self

    def vline(self, x: float, color: str = "var(--text-3)", dash: Optional[str] = "3 4",
              width: float = 1.2, y0: Optional[float] = None, y1: Optional[float] = None) -> "Fig":
        px = float(self.X(x))
        a = self.h - self.b if y0 is None else float(self.Y(y0))
        b = self.t if y1 is None else float(self.Y(y1))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{a:.1f}" y2="{b:.1f}" '
                          f'stroke="{color}" stroke-width="{width}"{d}/>')
        return self

    def hline(self, y: float, color: str = "var(--text-3)", dash: Optional[str] = "3 4",
              width: float = 1.2) -> "Fig":
        py = float(self.Y(y))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{self.l}" x2="{self.w - self.r}" y1="{py:.1f}" y2="{py:.1f}" '
                          f'stroke="{color}" stroke-width="{width}"{d}/>')
        return self

    def band(self, x0: float, x1: float, color: str, opacity: float = 0.14) -> "Fig":
        a, b = float(self.X(x0)), float(self.X(x1))
        self.parts.append(f'<rect x="{min(a, b):.1f}" y="{self.t}" width="{abs(b - a):.1f}" '
                          f'height="{self.h - self.t - self.b}" fill="{color}" opacity="{opacity}"/>')
        return self

    def text(self, x: float, y: float, s: str, color: str = "var(--text)", anchor: str = "start",
             dx: float = 0, dy: float = 0, weight: int = 600, size: float = 13, halo: bool = True) -> "Fig":
        px, py = float(self.X(x)) + dx, float(self.Y(y)) + dy
        cls = "lab halo" if halo else "lab"
        self.parts.append(f'<text class="{cls}" x="{px:.1f}" y="{py:.1f}" text-anchor="{anchor}" '
                          f'fill="{color}" font-weight="{weight}" font-size="{size}">{html.escape(s)}</text>')
        return self

    def text_px(self, px: float, py: float, s: str, color: str = "var(--text-2)",
                anchor: str = "start", weight: int = 500, size: float = 12.5) -> "Fig":
        self.parts.append(f'<text class="lab" x="{px:.1f}" y="{py:.1f}" text-anchor="{anchor}" '
                          f'fill="{color}" font-weight="{weight}" font-size="{size}">{html.escape(s)}</text>')
        return self

    def raw(self, svg: str) -> "Fig":
        self.parts.append(svg)
        return self

    def svg(self, title: str, desc: str = "") -> str:
        """Finished figure. `title` and `desc` are read by screen readers."""
        return (f'<svg class="fig" viewBox="0 0 {self.w} {self.h}" role="img" '
                f'aria-label="{html.escape(title)}" preserveAspectRatio="xMidYMid meet">'
                f'<title>{html.escape(title)}</title>'
                f'<clipPath id="{self.clip}"><rect x="{self.l}" y="{self.t - 2}" '
                f'width="{self.w - self.l - self.r}" height="{self.h - self.t - self.b + 4}"/></clipPath>'
                + (f'<desc>{html.escape(desc)}</desc>' if desc else "")
                + "".join(self.parts) + '</svg>')


def polygon_px(points, fill: str = "none", stroke: str = "none", width: float = 1.5,
               opacity: float = 1.0, dash: Optional[str] = None) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
            f'opacity="{opacity}" stroke-linejoin="round"{d}/>')


def legend(items: Sequence[tuple[str, str, str]]) -> str:
    """HTML legend row. items: (label, colour, kind) with kind in line|dash|dot|box."""
    out = []
    for label, colour, kind in items:
        if kind == "dot":
            mark = f'<i class="k-dot" style="background:{colour}"></i>'
        elif kind == "box":
            mark = f'<i class="k-box" style="background:{colour}"></i>'
        elif kind == "dash":
            mark = f'<i class="k-line" style="border-top:2px dashed {colour}"></i>'
        else:
            mark = f'<i class="k-line" style="border-top:2.5px solid {colour}"></i>'
        out.append(f'<span>{mark}{html.escape(label)}</span>')
    return '<div class="legend">' + "".join(out) + '</div>'
