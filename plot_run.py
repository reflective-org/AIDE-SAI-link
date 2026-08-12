"""Figures for the 90-day coupled SAI run (physical TOMAS + radiation).

Same three figures the retired viz_coupled_month.py drew, but self-contained and it writes
{TAG}_*.png DIRECTLY -- no week_*.png intermediate and no mv step in a watchdog.
That indirection was the source of "I edited the script and nothing changed":
the edit landed in week_sizedist.png while zonal90d_sizedist.png stayed stale.

  1) dashboard      -> {TAG}_dashboard.png   burdens, radiative feedback, size, gases, budget
  2) filmstrip      -> {TAG}_filmstrip.png   dT_rad + aerosol mass at the probe level
  3) size-dist      -> {TAG}_sizedist.png    dN/dlogDp evolution, two panels
                                             (global mean | 15S-15N), fixed axes

  python3 plot_run.py [TAG]        # TAG defaults to zonal90d
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

TAG = sys.argv[1] if len(sys.argv) > 1 else "zonal90d"

# fail with the list of real tags instead of a bare FileNotFoundError on a
# hardcoded default that may not exist any more
if not os.path.exists(f"coupled_timeseries_{TAG}.npz"):
    have = sorted(g[len("coupled_timeseries_"):-len(".npz")]
                  for g in glob.glob("coupled_timeseries_*.npz")
                  if not g.endswith("_ckpt.npz"))
    sys.exit(f"no run named '{TAG}'.\navailable tags:\n  " + "\n  ".join(have))

ts = np.load(f"coupled_timeseries_{TAG}.npz")
# The frames file is the fragile input: it is the biggest, it is written last, and
# a truncated one fails HERE in the NpzFile constructor -- a zip's central
# directory lives at the END of the file, so np.load cannot even open it, let
# alone lazily fail on first member access. Guard the open itself, or one bad
# frames file costs all three figures including the dashboard, which needs
# nothing but the timeseries. (A coupled_frames_*.CORRUPT.npz has happened.)
try:
    fr = np.load(f"coupled_frames_{TAG}.npz")
except Exception as _e:
    print(f"  frames file unreadable ({type(_e).__name__}: {_e})\n"
          f"  -> dashboard only; no filmstrip, no size-distribution", flush=True)
    fr = None

RHO_AER = 1770.0                       # kg/m3 sulfate (for mass<->diameter)

# ---- burden units -------------------------------------------------------
# coupling.py's "burden" is sum over the slab of q * dp * <cos(phi)>, i.e. a
# mixing ratio (kg/kg for mass, #/kg for number) times Pa, with the per-cell
# area factor R^2 * dphi * dlambda / g DIVIDED OUT (it is a constant on this
# uniform f09 grid, so it cancels in every ratio the run reports). Multiply it
# back in and the burden becomes kg (or particles):
#     kg = burden * R^2 * dphi * dlambda / g
# Cross-checked against the run's own injection accounting: 10 Tg SO2/yr for
# 90 days = 2.46575 Tg, and injSO2_cum ends at 1.660146 -> 1.485246 Tg per
# burden unit, which is BURDEN_TG below to 6 digits.
R_EARTH, GRAV = 6.371e6, 9.80665
NLAT_G, NLON_G = 192, 288
BURDEN_KG = R_EARTH ** 2 * (np.pi / (NLAT_G - 1)) * (2.0 * np.pi / NLON_G) / GRAV
BURDEN_TG = BURDEN_KG / 1.0e9          # Tg of tracer per unit of (kg/kg * Pa)
# the aerosol mass tracer is SO4-mass (the fast model's condensation is
# S-conserving: H2SO4 kg -> SO4 kg * 96/98), so Tg SO4 -> Tg S is 32.06/96.06
S_PER_SO4 = 32.06 / 96.06
S_PER_SO2 = 32.06 / 64.06
# accessible qualitative colors
C = dict(num="#d1495b", mass="#00798c", dT="#d1495b", rms="#edae49", arf="#8f2d56",
         a="#2e4057", b="#66a182", so2="#e07a5f", h2so4="#3d5a80")

# ---- frame geometry, shared by all three figures ------------------------
# Bin diameters, frame times and the latitude metric are needed by BOTH the
# dashboard's size panel and the size-distribution figure, so they are derived
# once here instead of inside FIG 3.
# nf comes from frame_hours, which every frames file has, NOT from frames_dT:
# deriving it from a filmstrip-only array left it undefined whenever the
# filmstrip was skipped.
#
# GUARDED, because this block runs before anything is drawn: the frames file is
# the one input here that is routinely absent, truncated or mid-write (a
# coupled_frames_*.CORRUPT.npz is a thing that has happened), and the dashboard
# is built to render off the timeseries alone in exactly that case -- see the
# probe_hpa try/except below. Reading frames unguarded up here took all three
# figures down instead of the two that actually need frames.
try:
    if fr is None:                                    # already reported at the open
        raise RuntimeError("frames file could not be opened")
    fh = np.asarray(fr["frame_hours"], float)
    nf = len(fh)
    xk = np.asarray(fr["xk"])                         # (41) bin mass boundaries [kg]
    Dp_edge = (6.0 * xk / (np.pi * RHO_AER)) ** (1.0 / 3.0) * 1e9     # nm
    Dp_mid = np.sqrt(Dp_edge[:-1] * Dp_edge[1:])                     # (40)
    dlogDp = np.log10(Dp_edge[1:] / Dp_edge[:-1])                    # (40)
    FRAMES_OK = True
except Exception as _e:
    if fr is not None:                                # a file that opened but is
        print(f"  frame geometry unusable ({type(_e).__name__}: {_e})\n"
              f"  -> dashboard only; no filmstrip, no size-distribution", flush=True)
    fh = np.zeros(0); nf = 0; Dp_mid = dlogDp = None
    FRAMES_OK = False

_NUM = None
_NUM_READ = False        # a separate flag, not a sentinel value in _NUM: once
                         # _NUM holds the array, `_NUM == <sentinel>` is an
                         # elementwise compare and raises on the truth test


def get_num():
    """frames_num (nf,40,nlat,nlon) [#/kg], read once, or None if unavailable.

    NOT bound at import: this is ~1.6 GB for a 90-day run and ~5 GB for a year,
    and a dashboard for a run that recorded reff_nm never touches it. Read once
    and held, though -- an NpzFile member re-reads from disk on EVERY
    __getitem__, so indexing fr["frames_num"] in a loop re-reads the whole array
    per iteration.
    """
    global _NUM, _NUM_READ
    if not _NUM_READ:
        _NUM_READ = True
        try:
            _NUM = fr["frames_num"] if "frames_num" in fr.files else None
        except Exception as e:
            print(f"  frames_num unreadable ({type(e).__name__}: {e})", flush=True)
            _NUM = None
    return _NUM


_LATF = None


def lat_full():
    """Frame latitudes [deg], from the run when it stored them, else the f09 grid."""
    global _LATF
    if _LATF is None:
        if "lat" in fr.files:
            _LATF = np.asarray(fr["lat"])
        else:
            _n = get_num()
            _LATF = np.linspace(-90, 90, _n.shape[2] if _n is not None else 192)
    return _LATF


def wlat_full():
    """cos(lat) area metric for every horizontal mean here, built once.

    A plain .mean() over (lat,lon) weights every grid cell equally, which on a
    regular lat-lon grid over-counts the poles (many cells, little area) and read
    0.63x the true global mean at the ultrafine peak.
    """
    return np.cos(np.deg2rad(lat_full()))

hrs = np.asarray(ts["hours"], float); days = hrs / 24.0
N0 = float(ts["N0"]); M0 = float(ts["M0"])
N_DAYS_RUN = days[-1]                                  # actual run duration, for titles
# samples per day -- nucleation is OH-driven so anything number-related carries a
# strong diurnal cycle that aliases against the 6h sampling
# A 1-step run has nothing to diff, so the median is NaN and int(NaN) raises --
# which made a short smoke run (the first thing anyone does on a new machine)
# crash here instead of plotting. Fall back to the step length the run actually
# used, then to the 6h production default.
_dh = np.diff(hrs)
_dh = _dh[np.isfinite(_dh) & (_dh > 0)]
_step_h = float(np.median(_dh)) if _dh.size else (float(hrs[0]) if hrs[0] > 0 else 6.0)
PER_DAY = max(1, int(round(24.0 / _step_h)))


def running_mean(y, w):
    """Centered running mean. The window shrinks at the ends (divide by the
    actual number of samples covered) so the endpoints are not dragged toward
    zero the way a zero-padded convolution would drag them."""
    y = np.asarray(y, float)
    # np.convolve(..., "same") returns max(len(y), w) samples, NOT len(y), so a
    # window wider than the series silently LENGTHENS it and the caller then
    # plots y against a shorter x. Clamp instead: on a run shorter than the
    # averaging window there is nothing to smooth anyway.
    w = max(1, min(int(w), y.size))
    k = np.ones(w)
    return np.convolve(y, k, "same") / np.convolve(np.ones_like(y), k, "same")


def add_headroom(ax, frac=0.22, where="top"):
    """Open an empty strip at the top (or bottom) of ax's data range.

    Every dashboard panel here has two crossing curves that between them touch
    all four corners, so "move the legend to a free corner" has no answer -- the
    only way to get a box off the data is to make room for it. Log axes are
    stretched by a fraction of their DECADE range, not of hi-lo, or the strip
    collapses to nothing on (a). Call this after everything is plotted.
    """
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log" and lo > 0:
        r = (hi / lo) ** frac
        ax.set_ylim((lo, hi * r) if where == "top" else (lo / r, hi))
    else:
        d = frac * (hi - lo)
        ax.set_ylim((lo, hi + d) if where == "top" else (lo - d, hi))


def save(fig, name, **kw):
    """Write {TAG}_{name}.png and say where it went."""
    path = f"{TAG}_{name}.png"
    fig.savefig(path, dpi=125, **kw)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


# =====================================================================
# FIG 1 -- time-evolution dashboard
# =====================================================================
# 2x2, one panel per function. It was 2x3 until two panels were dropped outright:
#   * dT_rad max, which was half of the old (b). ARF_toa was the other half and it
#     moved onto the AOD panel, where it belongs anyway -- AOD and TOA forcing are
#     the same quantity seen through the optics and through the flux, so plotting
#     them on one pair of axes makes the near-mirror relation visible instead of
#     leaving the reader to eyeball it across two panels.
#   * gas-phase burdens (the old (e)). See the git history if you need it back:
#     nothing else on the figure depends on it. Its H2SO4 curve was also the most
#     misread thing here -- its first ~20 days are the CESM upper-stratospheric
#     vapor reservoir draining, not an SAI signal.
# dT_rms was never plotted: runs before the area-weighting fix wrote it from a
# plain unweighted .mean() over (nlev,nlat,nlon), ~20% low (0.616 vs 0.769 K at
# day 203 of prod1yr), and only the scalar was stored so it cannot be recovered.

# (a) burdens: N/N0 (log) + M/M0 (linear)
# NOTE ON THE TITLE: this used to read "number runaway vs mass conservation",
# which is wrong on both counts for an open-system SAI run -- number turns over
# once the condensation sink kills nucleation, and mass is SUPPOSED to grow
# (~4x here) because we inject continuously. Do not put "conservation" back.
# The raw N/N0 swings ~+/-25% on the diurnal nucleation cycle, which is wider
# than the trend it hides, so plot the 24h running mean as the signal and the
# raw samples faintly behind it -- same treatment arf_toa already gets.
def p_burden(ax, letter="a"):
    nrat = ts["Nburden"] / N0
    # the faint raw samples get NO legend entry -- a thin band behind a smoothed
    # curve reads as variability on sight, and naming it cost a third of the box
    ax.plot(days, nrat, "-", color=C["num"], lw=0.6, alpha=0.30, label="_nolegend_")
    ax.plot(days, running_mean(nrat, PER_DAY), "o-", color=C["num"], ms=3,
            label="N / N0  (24h mean)")
    ax.set_yscale("log"); ax.set_ylabel("N / N0  [dimensionless]  (log)", color=C["num"])
    ax.tick_params(axis="y", labelcolor=C["num"])
    ax2 = ax.twinx()
    ax2.plot(days, ts["Mburden"] / M0, "s-", color=C["mass"], ms=3, label="M / M0  (mass)")
    ax2.set_ylabel("M / M0  [dimensionless]  (linear)", color=C["mass"])
    ax2.tick_params(axis="y", labelcolor=C["mass"])
    ax2.axhline(1.0, color=C["mass"], lw=0.6, ls=":")
    # both curves are ratios, so the panel carries no scale of its own -- state the
    # normalizers in absolute units (see BURDEN_TG) rather than leaving the reader
    # to guess what "M / M0 = 1.9" is 1.9 of
    ax.text(0.03, 0.97,
            f"N0 = {N0 * BURDEN_KG:.2e} particles\n"
            f"M0 = {M0 * BURDEN_TG:.3f} Tg SO4 ({M0 * BURDEN_TG * S_PER_SO4:.3f} Tg S)",
            transform=ax.transAxes, va="top", ha="left", fontsize=7, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="w", ec="0.8", alpha=0.9))
    ax.set_title(f"({letter}) burden: number turnover vs injection-driven mass gain")
    ax.set_xlabel("day")
    # M/M0 is on the twin axis, so a plain ax.legend() left it out entirely -- merge
    # both axes' handles or the panel's second curve goes unnamed
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=7, framealpha=0.85)
    ax.grid(alpha=0.25)

# (b) AOD550 + TOA forcing, the optical and the flux view of the same aerosol
def p_aod(ax, letter="b"):
    ax.plot(days, ts["aod550"], "o-", color=C["a"], label="AOD550")
    ax.set_ylabel("AOD550 [dimensionless]", color=C["a"])
    ax.tick_params(axis="y", labelcolor=C["a"])
    ax.set_xlabel("day")
    axb = ax.twinx()
    # arf_toa is an instantaneous single-solar-time sample and sawtooths at the
    # sampling rate; arf_toa_avg is its trailing diurnal mean (see ARF_AVG_H). Plot
    # the mean as the signal and the raw samples faintly behind it -- unnamed, same
    # as the raw N/N0 band in (a): a faint band behind a smoothed curve is legible
    # as variability without a legend entry spent on it.
    if "arf_toa_avg" in ts.files:
        axb.plot(days, ts["arf_toa"], "-", color=C["arf"], lw=0.6, alpha=0.35,
                 label="_nolegend_")
        axb.plot(days, ts["arf_toa_avg"], "^-", color=C["arf"], ms=4,
                 label="ARF_toa (24h mean)")
    else:
        axb.plot(days, ts["arf_toa"], "^-", color=C["arf"], label="ARF_toa")
    axb.axhline(0, color=C["arf"], lw=0.5, ls=":")
    axb.set_ylabel("ARF_toa [W/m2]", color=C["arf"])
    axb.tick_params(axis="y", labelcolor=C["arf"])
    ax.set_title(f"({letter}) global-mean AOD (550 nm) + TOA forcing")
    # AOD rises left-to-right and ARF falls from 0 at the top left, so the two of
    # them cross the whole panel and every corner is on a curve -- loc="best" has no
    # good answer here. Merge both axes' handles, open a strip above the data on
    # BOTH axes (ARF's top is 0 by construction, so it needs the same lift), and pin
    # the box there in two columns.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    add_headroom(ax, 0.24); add_headroom(axb, 0.24)
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, fontsize=7,
              framealpha=0.85, borderaxespad=0.4)
    ax.grid(alpha=0.25)

# (c) mean particle size -- EFFECTIVE DIAMETER, and nothing else.
# Every other mean size this panel has carried is gone. In order of removal:
#   * number-mean Dp (meanDp_num_nm), 3-5 nm, pure nucleation-mode bookkeeping
#   * D(M/N) (meanDp_nm), the diameter of the mean-mass particle -- NOT a
#     mass-weighted diameter, despite once being labeled one. A nucleation mode
#     adds enormous NUMBER at almost zero mass and drags it down even while all
#     the mass moves to larger sizes; that mislabel is what produced the wrong
#     "injected sulfur does not grow the accumulation mode" conclusion.
#   * the mass-weighted diameter M_4/M_3 (meanDp_mass_nm), which was the
#     surviving reference curve. It is a defensible average of the distribution
#     but it is not the one the optics integrates, and having two size curves on
#     one axis (at one point one as a radius and one as a diameter) is exactly
#     how a factor of 2 gets read as physics. coupling.py still RECORDS it.
# What is left is D_eff = <D^3>/<D^2>, the third-over-second NUMBER moment: the
# mean size that reproduces the layer's optics, since radiation.py builds its Mie
# tables on the bin diameters and weights them by per-bin NUMBER. It is also the
# best conditioned of the four -- dropping the unconverged sub-10 nm bins moves
# it 2-3% in the mature phase against 15x for the number-mean.
# Reported as a DIAMETER (= 2*reff_nm) to match this axis' units; the SAI
# literature's r_eff ~ 0.4-0.5 um is half of what is plotted here.
# probe level for the (d) title. Guarded: npz members load lazily, so this is a
# few bytes off the frames file, but keep the dashboard renderable (e.g. off a
# live _ckpt timeseries) even when frames are absent or mid-write.
try:
    PROBE_HPA_LBL = f"{float(fr['probe_hpa']):.1f} hPa"
except Exception:
    PROBE_HPA_LBL = "probe level"

_HAS_REFF = "reff_nm" in ts.files


def effective_diameter():
    """(x_days, D_eff[nm], legend label), or None if this run has neither input.

    Computed ONCE at import: the frames branch reduces a ~5 GB array, and the
    panel is drawn into more than one figure.
    """
    # The label says only which of the two it is. The definition
    # (<D^3>/<D^2>, area-weighted moments, probe level) is in the axis label, the
    # title and this comment -- it does not need to be on the curve as well.
    if _HAS_REFF:
        # coupling.py's r_eff, at the full timeseries cadence. It is a ratio of
        # AREA-WEIGHTED moments on the WET droplet diameter (WET_OPTICS), i.e. the
        # same size the Mie tables are built on. Returns BEFORE get_num(), so a
        # run that recorded reff_nm never pays for the frames read at all.
        return days, 2.0 * np.asarray(ts["reff_nm"], float), "wet effective diameter"
    if not FRAMES_OK:
        return None
    num = get_num()
    if num is None:
        return None
    # Runs from before reff_nm was recorded (prod90d, prod1yr): rebuild the same
    # moment ratio from the per-bin frames at the probe level rather than falling
    # back to a different size diagnostic. Two differences: it is at FRAME cadence
    # (daily, not 6-hourly), and it is DRY.
    # WHY DRY, since it matters by 10-50%: Dp_mid here is bit-for-bit coupling.py's
    # DP_BIN -- cbrt(MMID/RHO_AER * 6/pi) on the geometric-mean bin mass, with
    # RHO_AER = 1770 kg/m3, the density of pure sulfate. No water enters, and the
    # frames carry only per-bin number, so no water COULD enter (the wet size needs
    # the local T/RH, which is not in the frames file). It is also the right
    # convention for these runs: they predate WET_OPTICS, so their own stored size
    # diagnostics were dry too -- cross-checked by recomputing meanDp_mass from the
    # frames (930 nm) against the value the run stored (891 nm). Dry recomputation
    # lands 4% ABOVE the stored number (an area-weighting detail); a wet diameter
    # would have to land well BELOW it, since the growth factor only ever adds water.
    # Ratio of area-weighted moments, not a mean of per-cell ratios: the layer's
    # optical depth is set by the summed moments, so the ratio of the sums is the
    # size that reproduces it. A mean of ratios would give a near-empty polar cell
    # the same say as a plume cell.
    wn = (num * wlat_full()[None, None, :, None]).sum(axis=(2, 3))   # (nf,40)
    deff = ((wn * Dp_mid ** 3).sum(axis=1)
            / np.maximum((wn * Dp_mid ** 2).sum(axis=1), 1e-300))
    return fh / 24.0, deff, "dry effective diameter"


DEFF = effective_diameter()


def p_size(ax, letter="c"):
    if DEFF is None:
        ax.text(0.5, 0.5, "no reff_nm in the timeseries and no\nframes_num to "
                "rebuild it from", transform=ax.transAxes, ha="center",
                va="center", fontsize=9)
    else:
        _x, _y, _lab = DEFF
        ax.plot(_x, _y, "o-", color=C["b"], ms=3.5, lw=1.8, label=_lab)
    # NOTE: unlike the burden and budget panels, which are full-slab, this one is a
    # SINGLE LEVEL -- the KPROBE probe level (51.7 hPa, the injection level). Say so
    # in the title; it read as a slab quantity purely from its company on the figure.
    ax.set_title(f"({letter}) effective diameter of the aerosol  @ {PROBE_HPA_LBL}",
                 fontsize=10)
    ax.set_xlabel("day")
    ax.set_ylabel("effective diameter $D_{eff}$ [nm]")
    # one curve now, but it is U-shaped (nucleation burst down, then condensational
    # growth back up), so it comes close to both upper corners -- open a strip rather
    # than trusting loc="best" to find one
    add_headroom(ax, 0.20)
    if DEFF is not None:
        ax.legend(loc="upper center", fontsize=7.5, framealpha=0.85, borderaxespad=0.4)
    ax.grid(alpha=0.25)

# (d) cumulative mass budget by stage (dM/M0)
# NOTE: coupling.py stores B_* as ALREADY-cumulative running totals (cumB[k]),
# so these are plotted as-is. Taking np.cumsum here double-integrates them --
# that bug made the panel read micro +38 / advect -22 when the run's own budget
# line said +0.88 / -0.34.
# Only two stages set the scale: micro (+4.3) and advect (-1.1). settle, floor,
# adv_pol and bc are ~100x smaller -- plotting them here (on either axis) buys
# four unreadable lines and a six-entry legend, so their end values go in a text
# box instead. They ARE still summed into the closure check: B_floor and
# B_adv_pol were left out of the old version entirely, so its lines did not add
# up to M/M0-1 (floor is in fact bigger than settle).
# single axis, in absolute Tg SO4. The B_* are stored as fractions of M0, so
# multiply through by M0 here; the dimensionless dM/M0 axis this panel used to
# carry (with Tg on a secondary_yaxis) is gone -- Tg is the unit the stage terms
# are actually compared in, and one axis cannot be misread for the other.
MTG = M0 * BURDEN_TG                       # Tg SO4 per unit of dM/M0
BIG = [("B_micro", "micro (source)", C["num"]), ("B_adv_np", "advect (sink)", C["a"])]
SMALL = [("B_settle", "settle", C["b"]), ("B_floor", "floor", C["so2"]),
         ("B_adv_pol", "polar", C["h2so4"]), ("B_bc", "bc", C["arf"])]
_have = [k for k, _, _ in BIG + SMALL if k in ts.files]
bsum = np.sum([ts[k] for k in _have], axis=0)
resid = bsum[-1] - (ts["Mburden"][-1] / M0 - 1.0)


def p_budget(ax, letter="d"):
    for key, lab, col in BIG:
        if key in ts.files:
            ax.plot(days, ts[key] * MTG, "-", color=col, lw=1.8, label=lab)
    ax.plot(days, (ts["Mburden"] / M0 - 1.0) * MTG, "--", color="0.25", lw=1.4,
            label="net (M - M0)")
    ax.axhline(0, color="0.6", lw=0.6)
    # small terms in Tg to match the axis; they are ~100x smaller than micro/advect
    # so they stay in the box rather than becoming four unreadable lines
    small_txt = "\n".join(f"{lab:>7s} {ts[key][-1] * MTG:+.4f}"
                          for key, lab, _ in SMALL if key in ts.files)
    ax.text(0.03, 0.03,
            f"small terms (end) [Tg SO4]:\n{small_txt}\n"
            f"{'closure':>7s} {resid * MTG:+.0e}\n{'M0':>7s} {MTG:.3f}",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=7,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="w", ec="0.8", alpha=0.9))
    ax.set_title(f"({letter}) cumulative mass budget by stage"); ax.set_xlabel("day")
    ax.set_ylabel("cumulative dM [Tg SO4]")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    # the small-terms box lives at the bottom-left and the advect curve runs right
    # through where it lands once the axis is in Tg (a ~3.5 Tg span instead of the
    # old ~5.5 dimensionless one). Open up room below the data rather than moving
    # the box: every other corner is occupied too (legend upper-left, micro
    # upper-right, advect lower-right).
    add_headroom(ax, 0.32, where="bottom")
    ax.grid(alpha=0.25)


print(f"  budget closure: sum {bsum[-1]:+.6f}  vs  M/M0-1 "
      f"{ts['Mburden'][-1]/M0-1:+.6f}  (residual {resid:+.2e})", flush=True)

# ---- assemble: 2x2, one call per panel ----
# figsize keeps the ~5.7x4.5in panel of the old 2x3 so the fonts stay the size
# they were tuned at; shrinking the canvas to 2 columns instead would make every
# legend and text box on the figure relatively larger.
fig, axs = plt.subplots(2, 2, figsize=(11.5, 9))
fig.suptitle(f"{N_DAYS_RUN:.0f}-day coupled run ({TAG}): closed aerosol-radiation-microphysics loop",
             fontsize=14, fontweight="bold")
p_burden(axs[0, 0], "a")
p_aod(axs[0, 1], "b")
p_size(axs[1, 0], "c")
p_budget(axs[1, 1], "d")
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "dashboard")

# =====================================================================
# FIG 2 -- spatial filmstrip: dT_rad + aerosol mass across the run
# =====================================================================
# Skip rather than die if this frames file has no dT/mass. The three figures need
# different arrays -- only the filmstrip needs frames_dT and frames_mas, while the
# size-dist below needs frames_num alone -- so a frames file carrying a subset
# should still produce every figure it CAN. Unconditionally indexing frames_dT
# here would abort the script before the size-dist ran, losing a figure whose
# inputs were fully present. (Concretely: on 2026-08-03 a truncated ckpt left 303
# frames of frames_num recoverable and nothing of frames_dT/frames_mas.)
nlat, nlon = 192, 288
lat = np.linspace(-90, 90, nlat); lon = np.linspace(0, 360, nlon, endpoint=False)
# fh / nf come from the shared frame-geometry block at the top (frame_hours, which
# every frames file has -- NOT frames_dT, which may be absent here). FRAMES_OK is
# false when that block could not read the file at all, which is a stronger
# failure than the per-array _missing check below.
_missing = [k for k in ("frames_dT", "frames_mas") if k not in fr.files] \
    if FRAMES_OK else ["*"]
if not FRAMES_OK:
    print(f"  filmstrip SKIPPED: {TAG} frames file is unreadable")
elif _missing:
    print(f"  filmstrip SKIPPED: {TAG} frames file has no {', '.join(_missing)}"
          f" (has {', '.join(sorted(fr.files))})")
else:
    probe = float(fr["probe_hpa"])
if not _missing:
    dT = fr["frames_dT"]                              # (nf,nlat,nlon)
    mass = fr["frames_mas"].sum(axis=1)               # (nf,nlat,nlon) total SO4 mass MR
    assert dT.shape[0] == nf, f"frames_dT has {dT.shape[0]} frames, frame_hours {nf}"
    sel = np.unique(np.linspace(0, nf - 1, min(5, nf)).astype(int))   # up to 5 snapshots

    proj = ccrs.PlateCarree()
    # PlateCarree axes are aspect-locked 2:1, so the figure height has to follow the
    # column width or the two rows float apart with a band of dead space between them
    COL_W = 4.2
    fig = plt.figure(figsize=(COL_W * len(sel), COL_W + 1.1), constrained_layout=True)
    fig.suptitle(f"Spatial evolution @ {probe:.1f} hPa: radiative heating (top) & aerosol mass (bottom)",
                 fontsize=13, fontweight="bold")
    # dT is strongly one-sided: ~-0.12 K of cooling against ~+3.9 K of heating.
    # Scale SYMMETRICALLY about 0 off the heating limb -- white stays at zero and
    # the cooling shows at its true (tiny) relative amplitude. Giving the negative
    # limb its own limits instead would stretch a 0.12 K range over half the
    # colormap and make trivial cooling look as strong as the heating.
    dpos = max(float(np.nanpercentile(dT[sel], 99.5)), 1e-6)
    dnorm = mcolors.Normalize(vmin=-dpos, vmax=dpos)
    mvmax = np.nanpercentile(mass[sel] * 1e9, 99.5)
    mnorm = mcolors.Normalize(0, mvmax)
    # keep the two rows in their own lists -- add_subplot appends to fig.axes in
    # call order (axt0, axm0, axt1, ...), so slicing fig.axes interleaves the rows
    # and anchors each colorbar to a full-height mix of both
    top_axes, bot_axes = [], []
    for j, fi in enumerate(sel):
        axt = fig.add_subplot(2, len(sel), j + 1, projection=proj)
        axt.pcolormesh(lon, lat, dT[fi], cmap="RdBu_r",
                       norm=dnorm, transform=proj, shading="auto")
        axt.coastlines(linewidth=0.3, color="0.3"); axt.set_title(f"day {fh[fi]/24:.1f}", fontsize=10)
        top_axes.append(axt)
        axm = fig.add_subplot(2, len(sel), len(sel) + j + 1, projection=proj)
        axm.pcolormesh(lon, lat, mass[fi] * 1e9, cmap="viridis", norm=mnorm,
                       transform=proj, shading="auto")
        axm.coastlines(linewidth=0.3, color="w")
        bot_axes.append(axm)
    sm1 = plt.cm.ScalarMappable(cmap="RdBu_r", norm=dnorm)
    sm2 = plt.cm.ScalarMappable(cmap="viridis", norm=mnorm)
    fig.colorbar(sm1, ax=top_axes, orientation="vertical", fraction=0.02, pad=0.01,
                 aspect=18, label="dT_rad [K]", extend="max")
    fig.colorbar(sm2, ax=bot_axes, orientation="vertical", fraction=0.02, pad=0.01,
                 aspect=18, label="SO4 [x1e-9 kg/kg]", extend="max")
    save(fig, "filmstrip", bbox_inches="tight")

# =====================================================================
# FIG 3 -- size-distribution evolution (dN/dlogDp) at the probe level
#   -> {TAG}_sizedist.png, two panels: (a) area-weighted GLOBAL mean,
#      (b) the same average restricted to 15S-15N.
# The tropics panel is the one to read for the injection region itself (the SAI
# source is a 15S-15N-ish zonal band, see INJ_ZONAL): the global mean dilutes it
# with extratropical air the plume has not reached yet. Both panels are drawn on
# IDENTICAL FIXED axes (SIZEDIST_XLIM/YLIM below, no autoscale on either end) --
# the whole point of the pair is reading the tropical enhancement off the panel
# offset, and a per-panel autoscale would silently rescale that away.
# =====================================================================
# ---- axis limits: edit these two. Both are hard limits shared by both panels;
# the y top is above the largest value either region reaches (~1.2e5 cm-3 STP,
# the tropical ultrafine end at day 0) with a little headroom.
SIZEDIST_XLIM = (10.0, 2000.0)     # dry diameter Dp [nm]
SIZEDIST_YLIM = (1e-1, 3e5)        # dN/dlogDp [cm-3 STP]

# Smallest bin to PLOT. Everything below this is dropped from the curves, not
# merely hidden by xlim, so the y-autoscale and the eye both ignore it.
# WHY 10 nm: the sub-10 nm population is not a converged model quantity. The
# operator-split A/B (PROCESSES.md 2.6) showed nucleation-mode number changes by
# ~60x purely from where condensation sits in the chain, while r_eff moves <1%
# and mass 0%. It also carries only ~0.04% of the mass. Plotting it invites the
# reader to treat 1.5e5 cm-3 STP of 3 nm particles at 20 km as a result; it is not.
DP_MIN_NM = 10.0

# Number is stored as a MIXING RATIO (# per kg air), so it must be multiplied by
# an air density to become a concentration. This plot -- like every other
# POSTPROCESSING diagnostic -- reports at STP, not at ambient density: an
# in-situ cm-3 at 51.7 hPa is ~15x smaller than the STP value and is not
# comparable against instrument data or across levels, whereas the STP number is
# just the mixing ratio in different clothes.
#
# The coupling itself is unaffected and must stay that way: the microphysics in
# coupling.py builds per-m3 concentrations from the LOCAL rho = p/(Rd*T) before
# calling the coagulation/nucleation/condensation kernels, because those rates
# are physically density-dependent. Ambient there, STP here.
#
# Consequence of the switch: the probe-level T no longer enters at all, so the
# former per-region reference temperatures (208.7 K global vs 202.9 K in the
# tropics, a 3% density difference that used to scale each panel separately) are
# gone. Both panels now share one constant, and the tropical enhancement the
# figure is built to show is pure aerosol, not partly a temperature contrast.
RHO_STP = 101325.0 / (287.05 * 273.15)   # kg/m3, dry air at 0 C / 1 atm

# Dp_mid / dlogDp come from the shared frame-geometry block at the top and the
# area metric from wlat_full() -- the dashboard's size panel needs the same three.
# Same reasoning as the filmstrip above: skip rather than die, so a frames file
# that cannot give this figure its inputs still leaves the other two standing.
_no_num = not FRAMES_OK or "frames_num" not in fr.files
if _no_num:
    print(f"  size-distribution SKIPPED: {TAG} frames file "
          + (f"has no frames_num (has {', '.join(sorted(fr.files))})" if FRAMES_OK
             else "is unreadable"))
if not _no_num:
    keep = Dp_mid >= DP_MIN_NM

    def dNdlogDp_of(latmask=None):
        """Area-weighted dN/dlogDp [cm-3 at STP] per frame, (nf,40), over the
        latitudes in latmask (None = all of them)."""
        # zero the weight outside the band instead of slicing, so the normalization
        # below (sum of weights actually used) stays a single expression
        _wf = wlat_full()
        wlat = _wf if latmask is None else np.where(latmask, _wf, 0.0)
        w = wlat[None, None, :, None]
        # get_num(), not fr["frames_num"]: this function is called once per panel and
        # an NpzFile member re-reads (1.6 GB here) on every access
        num = get_num()
        num_m = ((num * w).sum(axis=(2, 3))
                 / (w.sum() * num.shape[3]))            # (nf,40) regional mean, #/kg
        print(f"  sizedist: STP normalization, rho={RHO_STP:.5f} kg/m3 "
              f"(probe level is {float(fr['probe_hpa']):.1f} hPa); "
              f"plotted {int(keep.sum())}/{keep.size} bins, Dp >= {DP_MIN_NM:.0f} nm"
              + ("" if latmask is None else
                 f", {int(latmask.sum())}/{latmask.size} lat rows"), flush=True)
        return num_m * (RHO_STP / 1.0e6) / dlogDp[None, :]    # cm-3 STP per dlog10(Dp)

    # tropics: the injection band plus the ascending branch of the Brewer-Dobson
    # circulation, i.e. where the plume actually is for the first weeks
    TROPIC_LAT = 15.0
    PANELS = [("(a) global", dNdlogDp_of()),
              (f"(b) tropics ({TROPIC_LAT:.0f}S-{TROPIC_LAT:.0f}N)",
               dNdlogDp_of(latmask=np.abs(lat_full()) <= TROPIC_LAT))]

    cmap = plt.cm.viridis
    # sharey (not just equal set_ylim) so the right panel loses its tick labels too:
    # repeating them invites reading the panels as separately scaled
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.8), sharex=True, sharey=True)
    for ax, (label, dNdlogDp) in zip(axs, PANELS):
        for fi in range(nf):
            ax.plot(Dp_mid[keep], dNdlogDp[fi][keep],
                    color=cmap(fi / max(nf - 1, 1)), lw=1.5, alpha=0.9)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(*SIZEDIST_XLIM)
        ax.set_ylim(*SIZEDIST_YLIM)
        ax.set_xlabel("dry diameter Dp [nm]")
        ax.set_title(f"{label}  (area-weighted mean)", fontsize=11)
        ax.grid(alpha=0.25, which="both")
    axs[0].set_ylabel("dN/dlogDp  [cm$^{-3}$ at STP]")
    # Span the FRAMES, not N_DAYS_RUN. The colorbar below is already normalized to
    # fh[-1], so titling this with the timeseries duration mislabels any run whose
    # frames stop short of its last logged step -- e.g. a frames file recovered from
    # a truncated checkpoint, where the series reached day 334 but frames end at 302.
    _frame_days = fh[-1] / 24.0
    fig.suptitle(f"Particle-size distribution evolution ({_frame_days:.0f} days) "
                 f"@ {float(fr['probe_hpa']):.1f} hPa",
                 fontsize=12, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, fh[-1] / 24))
    fig.colorbar(sm, ax=axs, label="day", fraction=0.03, pad=0.11)
    # no tight_layout: it fights the colorbar's shared-axes placement. right= has to
    # leave room for BOTH the bar and its pad, or the bar is pushed off the canvas.
    fig.subplots_adjust(left=0.07, right=0.855, top=0.86, bottom=0.11, wspace=0.06)
    save(fig, "sizedist")

# ---- console summary ----
print("\n=== run summary ===", flush=True)
print(f"  duration: {days[-1]:.1f} days ({len(hrs)} steps), {nf} spatial frames", flush=True)
print(f"  N/N0: {ts['Nburden'][0]/N0:.1f} -> {ts['Nburden'][-1]/N0:.1f}  (number runaway)"
      f"   [{ts['Nburden'][0]*BURDEN_KG:.3e} -> {ts['Nburden'][-1]*BURDEN_KG:.3e} particles]",
      flush=True)
print(f"  M/M0: {ts['Mburden'][0]/M0:.4f} -> {ts['Mburden'][-1]/M0:.4f}  (mass)"
      f"   [{ts['Mburden'][0]*BURDEN_TG:.3f} -> {ts['Mburden'][-1]*BURDEN_TG:.3f} Tg SO4"
      f" = {ts['Mburden'][-1]*BURDEN_TG*S_PER_SO4:.3f} Tg S]", flush=True)
print(f"  SO2: {ts['SO2burden'][0]*BURDEN_TG:.4f} -> {ts['SO2burden'][-1]*BURDEN_TG:.4f} Tg SO2"
      f"   (injected {ts['injSO2_cum'][-1]*BURDEN_TG:.3f} Tg SO2 cumulative)", flush=True)
print(f"  H2SO4(g): {ts['H2SO4burden'][0]*BURDEN_TG:.3e} -> "
      f"{ts['H2SO4burden'][-1]*BURDEN_TG:.3e} Tg H2SO4", flush=True)
print(f"  dT_max: {ts['dT_max'][0]:.4f} -> {ts['dT_max'][-1]:.4f} K", flush=True)
_arfk = "arf_toa_avg" if "arf_toa_avg" in ts.files else "arf_toa"
print(f"  ARF_toa ({'24h mean' if _arfk.endswith('avg') else 'INSTANTANEOUS'}): "
      f"{ts[_arfk][0]:.4f} -> {ts[_arfk][-1]:.4f} W/m2", flush=True)
print(f"  AOD550: {ts['aod550'][0]:.4f} -> {ts['aod550'][-1]:.4f}  (dimensionless)", flush=True)
print(f"  meanDp(num): {ts['meanDp_num_nm'][0]:.1f} -> {ts['meanDp_num_nm'][-1]:.1f} nm", flush=True)
if DEFF is not None:
    _dv = DEFF[1]
    print(f"  D_eff{'' if _HAS_REFF else ' (from frames, dry)'}: "
          f"{_dv[0]:.1f} -> {_dv[-1]:.1f} nm  (r_eff {_dv[-1]/2e3:.3f} um)", flush=True)
