"""Charts for the dashboard, drawn with matplotlib.

Every function returns a `matplotlib.figure.Figure` built with the object API (no pyplot global state, so
concurrent Streamlit sessions cannot draw on each other's figures). Paces come in as seconds per km; the
`unit` argument only changes the axis labels. Colours: the coach's plan is dark ink, the model is dashed
blue, past races use three validated categorical colours, and green/amber mean ahead/behind.
"""
from __future__ import annotations

import io
import textwrap
from typing import Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Wedge
from matplotlib.ticker import FuncFormatter, MultipleLocator

from ..domain import MARATHON_KM, SEGMENT_KM
from ..planning import LABELS

INK, MUTED, GRID, LINE = "#18202F", "#5A6476", "#E6EAF1", "#C9D1DE"
BLUE, GREEN, AMBER = "#1D5BD8", "#137A55", "#9A5B00"
OVERLAY = ("#129C8D", "#D97A1E", "#7C5CC4")      # validated: CVD-separable, >= 3:1 on white
EDGES = np.concatenate([[0.0], np.cumsum(SEGMENT_KM)])
MILE = 1.609344


def _factor(unit: str) -> float:
    return MILE if unit == "mi" else 1.0


def _mmss(v: float, _pos=None) -> str:
    r = int(round(v))
    return f"{r // 60}:{r % 60:02d}"


def _clock(s: float) -> str:
    r = int(round(abs(s)))
    return f"{r // 3600}:{r % 3600 // 60:02d}:{r % 60:02d}" if r >= 3600 else f"{r // 60}:{r % 60:02d}"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def to_png(fig: Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def _steps(paces: Sequence[float], f: float) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for a, b, p in zip(EDGES[:-1], EDGES[1:], paces):
        xs += [a, b]
        ys += [p * f, p * f]
    return xs, ys


def pace_chart(plan: Sequence[float], model: Sequence[float], overlays: Sequence[tuple[str, Sequence[float]]] = (),
               unit: str = "km") -> Figure:
    """The pacing plan per interval: faster is higher. Interval widths are to scale (the last is 2.195 km)."""
    f = _factor(unit)
    fig = Figure(figsize=(8.6, 3.9), layout="constrained")
    ax = fig.add_subplot()
    _style(ax)
    series = [(name, paces, OVERLAY[k % 3], "-", 1.6) for k, (name, paces) in enumerate(overlays)]
    series += [("Model plan", model, BLUE, (0, (5, 3)), 1.8), ("Your plan", plan, INK, "-", 2.6)]
    for name, paces, color, ls, lw in series:
        xs, ys = _steps(paces, f)
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw, label=name, solid_capstyle="butt")
    mids = (EDGES[:-1] + EDGES[1:]) / 2
    ax.plot(mids, np.asarray(plan) * f, "o", color=INK, markersize=6, markerfacecolor="white", markeredgewidth=2)
    all_y = np.concatenate([np.asarray(p) * f for _, p, *_ in series])
    lo, hi = all_y.min(), all_y.max()
    pad = max(10.0, (hi - lo) * 0.15)
    ax.set_ylim(hi + pad, lo - pad)                       # inverted: faster paces at the top
    ax.yaxis.set_major_formatter(FuncFormatter(_mmss))
    ax.set_ylabel(f"Pace (min per {unit})  ·  faster ↑", color=MUTED, fontsize=9)
    ax.set_xlim(0, MARATHON_KM)
    ax.set_xticks(EDGES)
    ax.set_xticklabels([f"{e:g}" if e < 42 else "42.2" for e in EDGES])
    ax.set_xlabel("Distance (km)", color=MUTED, fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=len(series), frameon=False, fontsize=9,
              labelcolor=INK, handlelength=2.6)
    return fig


def race_bars(paces: Sequence[float], unit: str = "km", color: str = "#66C9A6", title: str = "") -> Figure:
    """Pace per 5 km interval as bars with labels, the layout of the paper's Figure 1."""
    f = _factor(unit)
    vals = np.asarray(paces) * f
    fig = Figure(figsize=(4.2, 2.5), layout="constrained")
    ax = fig.add_subplot()
    _style(ax)
    xs = np.arange(len(vals))
    ax.bar(xs, vals / 60, width=0.78, color=color)
    for x, v in zip(xs, vals):
        ax.text(x, v / 60 + 0.12, _mmss(v), ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_ylim(0, max(vals) / 60 + 1)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for lbl in LABELS], fontsize=6.5, rotation=0)
    ax.set_ylabel(f"Pace (min/{unit})", color=MUTED, fontsize=8)
    ax.set_xlabel("Distance interval (km)", color=MUTED, fontsize=8)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    return fig


def sparkline(paces: Sequence[float] | None) -> Figure:
    """A tiny pacing profile for a race card (steps to scale; higher is faster)."""
    fig = Figure(figsize=(2.4, 0.55))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    if not paces:
        ax.text(0, 0.5, "No 5 km splits", fontsize=8, color=MUTED, va="center")
        return fig
    xs, ys = _steps(paces, 1.0)
    ax.plot(xs, ys, color=INK, linewidth=1.4)
    ax.set_ylim(max(ys) + 5, min(ys) - 5)
    ax.set_xlim(0, MARATHON_KM)
    return fig


def range_chart(low: float, high: float, model: float, plan: float) -> Figure:
    """The likely range as a band, with the model's and the plan's finish times marked."""
    fig = Figure(figsize=(6.0, 1.15), layout="constrained")
    ax = fig.add_subplot()
    pad = max(120.0, (high - low) * 0.3)
    lo_lim, hi_lim = min(low, plan) - pad, max(high, plan) + pad
    ax.axhspan(-0.12, 0.12, color=GRID, zorder=0)
    ax.barh(0, high - low, left=low, height=0.42, color="#E8EFFC", edgecolor="#B9CCF3", zorder=1)
    ax.vlines(model, -0.36, 0.36, color=BLUE, linewidth=3, zorder=3)
    ax.vlines(plan, -0.36, 0.36, color=INK, linewidth=3, zorder=4)
    same = abs(plan - model) < (hi_lim - lo_lim) * 0.06
    ax.text(model, 0.48, f"Model {_clock(model)}", ha="right" if same else "center", va="bottom", fontsize=8.5, color=BLUE)
    ax.text(plan, -0.48, f"Your plan {_clock(plan)}", ha="left" if same else "center", va="top", fontsize=8.5, color=INK)
    ax.set_xlim(lo_lim, hi_lim)
    ax.set_ylim(-0.95, 0.95)
    ax.set_yticks([])
    ax.set_xticks([low, high])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: _clock(v)))
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return fig


def watch_face(st, unit: str = "km", runner: str = "") -> Figure:
    """A round watch screen for a `racewatch.WatchStatus`."""
    f = _factor(unit)
    fig = Figure(figsize=(3.6, 3.6))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.05, color="#2A2F3A"))          # bezel
    ax.add_patch(Circle((0, 0), 0.97, color="#0E1116"))          # screen
    ax.add_patch(Wedge((0, 0), 0.95, 0, 360, width=0.05, color="#2C3442"))
    ring = {"ahead": GREEN, "on": "#3D8BFF", "behind": "#E0A030", "slipping": "#E0A030", "done": "#3D8BFF",
            "start": "#3D8BFF"}[st.level]
    if st.progress > 0:
        ax.add_patch(Wedge((0, 0), 0.95, 90 - 360 * st.progress, 90, width=0.05, color=ring))
    white, dim = "#F5F7FA", "#9AA6B8"
    ax.text(0, 0.62, f"{st.distance_km:.2f} km", ha="center", va="center", fontsize=12, color=white, fontweight="bold")
    ax.text(0, 0.44, f"elapsed {_clock(st.elapsed_s)}", ha="center", va="center", fontsize=8.5, color=dim)
    if st.level == "done":
        ax.text(0, 0.12, _clock(st.elapsed_s), ha="center", va="center", fontsize=30, color=white, fontweight="bold")
        ax.text(0, -0.12, "finish time", ha="center", va="center", fontsize=8.5, color=dim)
    else:
        ax.text(0, 0.14, _mmss(st.target_pace_s * f), ha="center", va="center", fontsize=34, color=white, fontweight="bold")
        ax.text(0, -0.10, f"target min/{unit}", ha="center", va="center", fontsize=8.5, color=dim)
    if st.level == "start":
        delta_txt, delta_col = "ready", dim
    elif st.level == "done":
        word = "under plan" if st.delta_s < 0 else "over plan"
        delta_txt, delta_col = (f"{_clock(st.delta_s)} {word}" if round(st.delta_s) else "exactly on plan",
                                "#4CD39A" if st.delta_s <= 0 else "#F2B33D")
    elif abs(st.delta_s) <= 15:
        delta_txt, delta_col = "on plan", "#7FB2FF"
    elif st.delta_s < 0:
        delta_txt, delta_col = f"{_clock(st.delta_s)} ahead", "#4CD39A"
    else:
        delta_txt, delta_col = f"{_clock(st.delta_s)} behind", "#F2B33D"
    ax.text(0, -0.30, delta_txt, ha="center", va="center", fontsize=12, color=delta_col, fontweight="bold")
    if st.level != "done":
        ax.text(0, -0.50, f"{st.next_marker} target {_clock(st.next_marker_target_s)}", ha="center", va="center",
                fontsize=8.5, color=dim)
    ax.text(0, -0.70, runner[:22], ha="center", va="center", fontsize=8, color=dim)
    return fig


def ahead_behind_chart(track: Sequence[tuple[float, float]], paces: Sequence[float], upto_km: float | None = None) -> Figure:
    """Seconds ahead (above zero) or behind (below zero) plan over the race so far."""
    from ..racewatch import target_elapsed
    pts = [(d, t) for d, t in track if upto_km is None or d <= upto_km + 1e-9]
    fig = Figure(figsize=(8.6, 2.6), layout="constrained")
    ax = fig.add_subplot()
    _style(ax)
    ax.axhline(0, color=INK, linewidth=1)
    if len(pts) > 1:
        d = np.array([p[0] for p in pts])
        ahead = np.array([target_elapsed(paces, x) for x in d]) - np.array([p[1] for p in pts])
        ax.plot(d, ahead, color=INK, linewidth=2)
        ax.fill_between(d, ahead, 0, where=ahead >= 0, color=GREEN, alpha=0.18, interpolate=True, linewidth=0)
        ax.fill_between(d, ahead, 0, where=ahead < 0, color=AMBER, alpha=0.18, interpolate=True, linewidth=0)
        lim = max(30.0, float(np.abs(ahead).max()) * 1.2)
    else:
        lim = 60.0
    step = next((x for x in (10, 15, 30, 60, 120, 180, 300, 600, 900) if lim / x <= 4), 1800)
    lim = step * np.ceil(lim / step)
    ax.set_ylim(-lim, lim)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: ("+" if v > 0 else "−" if v < 0 else "") + _clock(v)))
    ax.set_xlim(0, MARATHON_KM)
    ax.set_xticks(EDGES)
    ax.set_xticklabels([f"{e:g}" if e < 42 else "42.2" for e in EDGES])
    ax.set_xlabel("Distance (km)", color=MUTED, fontsize=9)
    ax.set_ylabel("Ahead of plan (+) / behind (−)", color=MUTED, fontsize=9)
    return fig


def wrap(text: str, width: int = 60) -> str:
    return "\n".join(textwrap.wrap(text, width))
