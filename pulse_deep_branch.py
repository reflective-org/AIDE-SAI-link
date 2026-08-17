#!/usr/bin/env python3
"""The DEEP branch of the Brewer-Dobson circulation, as a tape recorder.

A lat-pressure snapshot shows WHERE the tracer got to. What it cannot show is
the deep branch's RATE, and rate is what makes the branch legible as a
circulation rather than as a stain spreading. This figure is the Hovmoller:

  x = calendar year (the runs start 1 Jan 1996, 365-day years), y = pressure,
  colour = that level's tropical (or polar) mean tracer NORMALISED TO ITS OWN
  MAX OVER TIME.

Per-level normalisation is the whole trick. Un-normalised, the tropical panel is
one bright blob at the source level and everything above it is invisible -- the
same fixed-scale problem that makes the top of {TAG}_pulse_progress_abs.png look
like it descends. Normalised, every level is on equal footing and the ascent
appears as a tilted RIDGE whose slope IS the residual vertical velocity.

  (a) tropics: one monotonic ridge, 43 hPa at the start -> 0.08 hPa 2.5 yr in.
      Fitted in two segments because w* accelerates upward -- printed, not drawn.
  (b) 60-80 N and (c) 60-80 S: where the branches ARRIVE, and it is not one ridge
      but two arrivals -- the lowermost stratosphere lights up after ~0.15 yr
      (shallow branch, straight off the subtropical jet) while the deep branch is
      still climbing the tropical pipe and does not reach the polar upper
      stratosphere until ~1.5 yr. That gap is the two-branch result. The caps are
      drawn separately because they run half a year out of phase, and because the
      Antarctic vortex makes them behave differently (see the comment on POLN).

  {TAG}_deep_branch.png

  python3 pulse_deep_branch.py [TAG] [--log]     # default pulse_15yr

Front convention: a level's ARRIVAL is the first time it reaches half its own
eventual peak. Half-of-own-peak, not time-of-peak: the peak is flat-topped and
noisy at levels the pulse only grazes, while the rising edge is sharp.

Palette: reversed+truncated magma -- multi-hue but MONOTONE IN LIGHTNESS, so it
reads as magnitude and survives greyscale, with far more mid-range contrast than
a single-hue ramp. NO OVERLAYS: no ridge lines, no slope fits, no altitude axis.
The shading shows the ascent and the descent on its own; every fitted number
prints to stdout. Do not put them back -- they were removed deliberately.
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# --log: identical figure, log colour scale instead of linear. Written to
# {TAG}_deep_branch_log.png so the canonical linear figure is not overwritten.
#
argv = [a for a in sys.argv[1:]]
LOG = "--log" in argv
if LOG:
    argv.remove("--log")

TAG = argv[0] if argv else "pulse_15yr"

C2 = "#eb6834"                      # accent: the arrival front
C3 = "#1baf7a"                      # accent: the descending limb
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8a86"
GRID = dict(color="0.85", lw=0.5, alpha=0.9)
H_KM = 7.5                          # scale height used for the z axis and w*
# Top of the plot. 1 hPa (~48 km), not the grid top at 0.049 hPa: above 1 hPa the
# air holds a fraction of a percent of the mass, so per-level normalisation makes
# that strip as dark as the core and it dominates a third of the panel height.
# The ascent fits below still run on the full column -- this crops the view, not
# the diagnostics.
P_TOP_PLOT = 1.0

# The runs start 1 Jan 1996 and the model calendar has 365-day years, so a day
# index maps exactly onto a calendar year. Every time AXIS and every printed time
# is in years; `days` survives only as the internal coordinate the rate fits are
# done in (km/day -> mm/s), never as a label.
YEAR0, DPY = 1996.0, 365.0


def to_year(d):
    return YEAR0 + np.asarray(d, float) / DPY

fr = np.load(f"coupled_frames_{TAG}.npz")
zm = np.asarray(fr["frames_zm_mas"], np.float64).sum(axis=1)   # (t, lev, lat)
lat = np.asarray(fr["lat"], float)
plev = np.asarray(fr["plev_hpa"], float)
dp = np.asarray(fr["dp_pa"], float)
days = np.asarray(fr["frame_hours"], float) / 24.0
yrs = to_year(days)                 # the x coordinate of every panel
Z_KM = H_KM * np.log(1013.0 / plev)


def hot_cmap():
    """Reversed, truncated magma -- multi-hue but MONOTONE IN LIGHTNESS, so it
    still encodes magnitude the way a single-hue ramp does and survives
    greyscale. Far more contrast than Blues across the mid-range, which is where
    the ascent ridge and the descending limb live."""
    return LinearSegmentedColormap.from_list(
        "hot", matplotlib.colormaps["magma"](np.linspace(0.98, 0.04, 256)), N=256)


CMAP = hot_cmap()


def region_mean(mask):
    """Mass-weighted mean mixing ratio per level over a latitude band."""
    w = (dp[:, None] * np.cos(np.deg2rad(lat)).clip(0.0)[None, :]) * mask[None, :]
    return np.einsum("tzy,zy->tz", zm, w) / w.sum(1)[None, :]


def arrival(q):
    """First day each level reaches half its own eventual peak."""
    out = np.full(q.shape[1], np.nan)
    for k in range(q.shape[1]):
        pk = q[:, k].max()
        if pk <= 0:
            continue
        hit = np.flatnonzero(q[:, k] >= 0.5 * pk)
        if hit.size:
            out[k] = days[hit[0]]
    return out


TROP = np.abs(lat) <= 15.0
# The two caps are drawn SEPARATELY. They were averaged together in the two-panel
# version because the descending branch follows the winter hemisphere -- the NH
# share of the extratropical exit swings 20% to 92% over the year -- so each cap
# on its own shows the descent as annual bursts, 6 months out of phase with the
# other, and a front fitted through one cap zigzags wherever a burst lands. That
# the split panels show the bursts and the hemispheric asymmetry that averaging
# the two caps together hides. The descent rate is fitted per cap below, from the
# annual-cycle phase lag, which is insensitive to which winter a burst lands in.
POLN = (lat >= 60.0) & (lat <= 80.0)
POLS = (lat <= -60.0) & (lat >= -80.0)
POLE = (np.abs(lat) >= 60.0) & (np.abs(lat) <= 80.0)

fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.8), constrained_layout=True,
                         sharey=True)
fig.suptitle(f"The deep branch, as a tape recorder  --  {TAG}, "
             f"{yrs[0]:.0f}-{yrs[-1]:.0f}, transport only",
             fontsize=13, fontweight="bold", color=INK)

panels = [
    (axes[0], TROP, "(a)  tropics, |lat| < 15"),
    (axes[1], POLN, "(b)  60-80 N"),
    (axes[2], POLS, "(c)  60-80 S"),
]

for ax, mask, title in panels:
    q = region_mean(mask)
    ref = q.max(0, keepdims=True)
    qn = q / np.where(ref > 0, ref, 1)
    if LOG:
        # clip=True so the (few) exact zeros and everything below the floor land
        # on the bottom colour rather than being masked out as white holes.
        pm = ax.pcolormesh(yrs, plev, qn.T, cmap=CMAP, shading="nearest",
                           norm=LogNorm(vmin=1e-4, vmax=1.0, clip=True))
    else:
        pm = ax.pcolormesh(yrs, plev, qn.T, cmap=CMAP, shading="nearest",
                           vmin=0, vmax=1)
    arr = arrival(q)
    ax.set_yscale("log")
    ax.set_ylim(plev.max(), P_TOP_PLOT)
    ax.set_yticks([100, 60, 30, 10, 6, 3, 1])
    ax.set_yticklabels(["100", "60", "30", "10", "6", "3", "1"])
    ax.set_xlim(yrs[0], yrs[-1])
    # one tick per calendar year, thinned when the run is long enough that every
    # label would collide
    _step = 1 if (yrs[-1] - yrs[0]) <= 8 else 2
    _tk = np.arange(np.ceil(yrs[0]), np.floor(yrs[-1]) + 1, _step)
    ax.set_xticks(_tk)
    ax.set_xticklabels([f"{t:.0f}" for t in _tk])
    ax.set_xlabel("year")
    ax.grid(True, **GRID)
    ax.set_title(title, fontsize=14, color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.panel_arrival = arr
axes[0].set_ylabel("pressure [hPa]")

# ---- fit the tropical ascent in two segments ------------------------------
# NOT DRAWN. The rates print to stdout instead -- the ridge is legible on its own
# and an overlay only argues with it. Two segments rather than one because w*
# accelerates with height in the tropical pipe; the break at 10 hPa is the
# conventional lower/middle stratosphere divide and close to where the slope
# visibly changes.
arr_t = axes[0].panel_arrival
seg = [(43.0, 10.0, "lower stratosphere"), (10.0, 0.4, "middle + upper")]
notes = []
for p_lo, p_hi, name in seg:
    m = (plev <= p_lo) & (plev >= p_hi) & np.isfinite(arr_t)
    if m.sum() < 3:
        continue
    # z = a*t + b, fitted on the ARRIVAL day, so `a` is the front's rise rate
    a, b = np.polyfit(arr_t[m], Z_KM[m], 1)
    w_mm_s = a * 1e6 / 86400.0                       # km/day -> mm/s
    notes.append((name, p_lo, p_hi, w_mm_s))

# ---- the DESCENDING limb ---------------------------------------------------
# Measured as the PHASE LAG of the annual cycle: each level's seasonal signal is
# isolated (divide by a 1-yr running mean), its annual harmonic phase is taken,
# the phases are unwrapped downward and z is regressed on them. The rate is the
# speed at which one winter's delivery propagates down the column.
#
# Two conventions were tried first and BOTH are unusable -- do not reinstate them:
#   * per-level day of maximum: hops between different winters' peaks and returns
#     +0.49 / -0.49 / -2.49 mm/s for N / S / combined, disagreeing in sign.
#   * first arrival after a fixed cutoff: every level below ~10 hPa is already
#     above half its post-cutoff maximum when the window opens, so the fit
#     measures the window edge and comes out with the wrong sign entirely.
# The phase lag uses the cycle REPEATING over 9 years rather than a single event,
# which is why it is stable (r = -0.94 N, -0.97 S) and correctly signed.
DESC_P = (3.0, 143.0)          # levels the phase is fitted over
_ds = (days > 3 * DPY) & (days < 12 * DPY)
_t = days[_ds]


def descent(mask):
    """Descent rate [mm/s] from the annual-cycle phase lag, with its fit r."""
    q = region_mean(mask)
    ph = []
    for k in range(len(plev)):
        y = q[:, k] / np.convolve(q[:, k], np.ones(73) / 73, "same")
        y = y[_ds]
        a1 = 2 * np.mean(y * np.cos(2 * np.pi * _t / DPY))
        b1 = 2 * np.mean(y * np.sin(2 * np.pi * _t / DPY))
        ph.append((np.degrees(np.arctan2(b1, a1)) / 360 * DPY) % DPY)
    sel = (plev >= DESC_P[0]) & (plev <= DESC_P[1])
    p_ = np.array(ph)[sel]
    for i in range(1, len(p_)):                 # unwrap: phase increases downward
        while p_[i] < p_[i - 1] - 30:
            p_[i] += DPY
    a = np.polyfit(p_, Z_KM[sel], 1)[0]
    return a * 1e6 / 86400.0, np.corrcoef(p_, Z_KM[sel])[0, 1], p_.max() - p_.min()


desc = [(nm, *descent(m)) for nm, m in [("60-80 N", POLN), ("60-80 S", POLS)]]
# ---- the polar arrivals, per cap ------------------------------------------
arrivals = [("60-80 N", axes[1].panel_arrival), ("60-80 S", axes[2].panel_arrival)]
_lo_m = plev >= 60.0
_hi_m = (plev <= 16.0) & (plev >= 1.0)
cb = fig.colorbar(pm, ax=axes, pad=0.055, aspect=30)
cb.set_label("tracer at that level / that level's own maximum over the run",
             fontsize=9)

p = f"{TAG}_deep_branch{'_log' if LOG else ''}.png"
fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Elapsed times print as years too. An arrival is a LAG from the start of the
# run, so it is quoted as a duration in years; the descent window is quoted as
# the calendar year it opens in, because that is a point on the x axis.
print(f"{TAG}: {len(days)} frames, {yrs[0]:.0f}-{yrs[-1]:.0f} "
      f"({days[-1] / DPY:.1f} yr)")
print("tropical ascent of the arrival front:")
for name, p_lo, p_hi, w in notes:
    print(f"  {name:<20s} {p_lo:6.1f} -> {p_hi:5.1f} hPa   w = {w:.2f} mm/s")
print("polar arrival (lag from the start of the run):")
for nm, a_p in arrivals:
    _l, _h = np.isfinite(a_p) & _lo_m, np.isfinite(a_p) & _hi_m
    print(f"  {nm:<14s} lowermost strat (>=60 hPa) {np.nanmedian(a_p[_l]) / DPY:.2f} yr"
          f"   upper strat (1-16 hPa) {np.nanmedian(a_p[_h]) / DPY:.2f} yr"
          f"   gap {(np.nanmedian(a_p[_h]) - np.nanmedian(a_p[_l])) / DPY:.2f} yr")
print(f"polar descending limb ({DESC_P[0]:.0f}-{DESC_P[1]:.0f} hPa, from the "
      f"annual-cycle phase lag; negative = descending):")
for nm, w, r, lag in desc:
    print(f"  {nm:<14s} {w:+.2f} mm/s   r={r:+.3f}   "
          f"top-to-bottom lag {lag:.0f} d over "
          f"{np.ptp(Z_KM[(plev >= DESC_P[0]) & (plev <= DESC_P[1])]):.1f} km")
print(f"wrote {p}", flush=True)
