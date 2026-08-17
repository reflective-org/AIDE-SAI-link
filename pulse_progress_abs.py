#!/usr/bin/env python3
"""Single-row snapshot version of the pulse progress figure.

Six cross-sections of tracer MASS per unit altitude on ONE SHARED ABSOLUTE
SCALE (everything divided by the initial peak), which is the only way to see the
decay: after 8 years the peak mass density is 0.2% of the initial peak, so on the
shared ramp the last panel is a faint smear and that faintness IS the result.

Panels are dated: the runs start 1 Jan 1996 and the model calendar has 365-day
years, so every label is a calendar month and year rather than a day index.

There used to be a companion row above this one with each panel scaled to its
OWN peak, to show the blob's SHAPE surviving the dilution. It is gone: the shape
locks in by the end of the first year, so the extra row was near-copies.

  {TAG}_pulse_progress_abs.png

Reads coupled_frames_{TAG}.npz from the CWD.

  python3 pulse_progress_abs.py [TAG]        # default pulse_15yr

The run length is read from the TAG ("pulse_8yr" -> 8 years), so a frames file
that stops short is labelled IN PROGRESS rather than silently retitled.

Palette: the sequential ramp is magma reversed and truncated -- multi-hue but
MONOTONE IN LIGHTNESS, so it encodes magnitude the way a single-hue ramp does
and survives greyscale. Accent blue is dataviz slot 1.
"""
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

TAG = sys.argv[1] if len(sys.argv) > 1 else "pulse_15yr"

C1 = "#2a78d6"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8a86"
GRID = dict(color="0.85", lw=0.5, alpha=0.9)

# The shared ramp of row (b) is DERIVED, not hardcoded, because its right limits
# depend on how far the run got: an 8-yr pulse ends 4x fainter than a 5-yr one,
# and the 5-yr floor of 1e-3 would put half the year-8 mass below the ramp.
# The rule -- ceiling at the SECOND panel's peak snapped to the nearest 1/3/10,
# floor two decades under it -- reproduces the values that were hand-tuned on
# pulse_5yr (0.1 and 1e-3) and rescales itself for any other run length.
#
# Anchoring the ceiling at the initial peak (1.0) instead would waste the top
# decade on a single panel and compress the late decay into two pale steps; the
# cost of the low ceiling is that the first panel saturates, which is flagged
# with an arrowhead on the colorbar and "(off scale)" in its title.
def snap13(x):
    """Nearest 1/3/10 x decade, in log space."""
    e = np.floor(np.log10(x))
    m = x / 10.0**e
    return float((1.0 if m < 10**0.25 else 3.0 if m < 10**0.75 else 10.0) * 10.0**e)


# The runs start 1 Jan 1996 and the model calendar has 365-day years, so a day
# index maps exactly onto a calendar date. Nothing user-facing is in days.
YEAR0, DPY = 1996, 365.0
_MON = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
_MNM = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def datestr(d):
    """Day index -> 'Mon YYYY' on the model's 365-day calendar."""
    y, doy = int(d // DPY), d % DPY
    return f"{_MNM[int(np.searchsorted(_MON, doy, side='right')) - 1]} {YEAR0 + y}"


fr = np.load(f"coupled_frames_{TAG}.npz")
zm = np.asarray(fr["frames_zm_mas"], np.float64).sum(axis=1)   # (t, lev, lat)
lat = np.asarray(fr["lat"], float)
plev = np.asarray(fr["plev_hpa"], float)
dp = np.asarray(fr["dp_pa"], float)
days = np.asarray(fr["frame_hours"], float) / 24.0

W = np.cos(np.deg2rad(lat)).clip(0.0)
WZY = dp[:, None] * W[None, :]
burden = np.einsum("tzy,zy->t", zm, WZY)

# Mass per unit ALTITUDE, not per model level: dz = H dp/p, so dM/dz ~ q*p*cos(lat).
# The y-axis is log-pressure (linear in altitude), so EQUAL AREAS HOLD EQUAL MASS.
mass_dens = zm * (plev[None, :, None] * W[None, None, :])
md0max = mass_dens[0].max()

# ONE PANEL PER YEAR: the initial state plus every 1 January the run reaches.
# Not a fixed count of evenly spaced frames -- a yearly cadence is what makes the
# decay readable as a rate, and it makes the panel grid the same shape for any
# run length.
want = [int(np.argmin(np.abs(days - DPY * k)))
        for k in range(int(days[-1] // DPY) + 1)]

# nominal run length from the tag, so a short frames file reads as IN PROGRESS
_yr = re.search(r"(\d+)yr", TAG)
total_days = int(_yr.group(1)) * 365 if _yr else int(round(days[-1]))

# The ramp has to span every panel now, not six. Ceiling at the year-1 peak
# (year 0 is the undiluted seed box and would waste a decade on one panel);
# floor a decade under the LAST panel's peak, so the final year still shows a
# blob rather than an empty frame. On the 15-yr run that is 1e-5 to 1e-1: four
# decades, against the two the six-panel version needed. Both ends are derived
# from the data, so any run length rescales itself.
VMAX = snap13(mass_dens[want[1]].max() / md0max)
VMIN = snap13(mass_dens[want[-1]].max() / md0max) / 10.0

# mass fraction of the LAST frame that survives the ramp floor -- printed, so a
# change to the ramp rule cannot leave a stale claim about what is being shown.
_vis = mass_dens[-1] / md0max > VMIN
f_shown = np.einsum("zy,zy->", zm[-1] * _vis, WZY) / burden[-1]

print(f"{TAG}: {len(days)} frames, {datestr(days[0])} - {datestr(days[-1])} "
      f"({days[-1] / DPY:.2f} of {total_days / DPY:.0f} yr)")
print(f"  mass remaining: {burden[-1] / burden[0]:.4f} of initial")
print(f"  ramp: {VMIN:.0e} - {VMAX:.0e} of the initial peak; the floor keeps "
      f"{f_shown:.1%} of the {datestr(days[-1])} mass, and the first panel "
      f"saturates (peak is {1 / VMAX:.0f}x the ceiling)")


def hot_cmap():
    return LinearSegmentedColormap.from_list(
        "hot", matplotlib.colormaps["magma"](np.linspace(0.93, 0.06, 256)), N=256)


def press_axis(ax):
    ax.set_yscale("log")
    ax.set_ylim(plev.max(), plev.min())
    ax.set_yticks([100, 30, 10, 3, 1, 0.3, 0.1])
    ax.set_yticklabels(["100", "30", "10", "3", "1", "0.3", "0.1"])
    ax.grid(True, **GRID)


cmap = hot_cmap()
cmap.set_bad("#eceff0")

# Grid, not a single row: at one panel per year a 15-yr run is 16 panels, and
# 16 across a page leaves each one too narrow to read a latitude off. NCOL=8
# keeps the reading order left-to-right within a row and puts a whole year on
# each column pair; the row break falls at the halfway point of the run.
NCOL = min(8, len(want))
NROW = int(np.ceil(len(want) / NCOL))
fig = plt.figure(figsize=(2.05 * NCOL, 3.55 * NROW), constrained_layout=True)
fig.suptitle(f"Tagged pulse: tropical lower stratosphere, 40-90 hPa, |lat|<15"
             f"  --  {YEAR0} to {YEAR0 + days[-1] / DPY:.0f}"
             f"  ({days[-1] / DPY:.1f} of {total_days / DPY:.0f} yr), one panel per year"
             + ("  (IN PROGRESS)" if days[-1] < total_days - 5 else ""),
             fontsize=13, fontweight="bold", color=INK)

# ---- one shared absolute scale: DECAY ------------------------------------
gs_b = fig.add_gridspec(NROW, NCOL, wspace=0.08, hspace=0.30)
ax_b = []
for j, i in enumerate(want):
    ax = fig.add_subplot(gs_b[j // NCOL, j % NCOL])
    ax_b.append(ax)
    f = np.ma.masked_less_equal(mass_dens[i] / md0max, VMIN)
    pm_b = ax.pcolormesh(lat, plev, f, cmap=cmap, shading="nearest",
                         norm=LogNorm(vmin=VMIN, vmax=VMAX))
    press_axis(ax)
    ax.set_xlim(-90, 90)
    ax.set_xticks([-60, 0, 60])
    ax.set_xticklabels(["60S", "EQ", "60N"], fontsize=8)
    # 1.05, not 1.0: the ceiling is SNAPPED to the second panel's peak, so that
    # panel can land a fraction of a percent over it. Flagging that as "off
    # scale" next to the first panel, which is over by 10-30x, reads as real clipping.
    over = "  (off scale)" if mass_dens[i].max() / md0max > 1.05 * VMAX else ""
    ax.set_title(f"{datestr(days[i])}{over}\n{burden[i] / burden[0]:.2%} of initial mass",
                 fontsize=9, color=INK)
    if j % NCOL == 0:
        ax.set_ylabel("pressure [hPa]")
    else:
        ax.set_yticklabels([])
cb = fig.colorbar(pm_b, ax=ax_b, pad=0.006, aspect=26, extend="max")
cb.set_label("mass density / the initial peak  --  one shared scale", fontsize=9)

# No caption block: the numbers that used to live there (the ramp limits, the
# mass fraction the floor keeps, the first panel's saturation) go to stdout instead, so
# the figure stays clean and the claims stay checkable.
p = f"{TAG}_pulse_progress_abs.png"
fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {p}", flush=True)
