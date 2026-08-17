"""Figures for the 90-day coupled SAI run (physical TOMAS + radiation).

Same three figures the retired viz_coupled_month.py drew, but self-contained and it writes
{TAG}_*.png DIRECTLY -- no week_*.png intermediate and no mv step in a watchdog.
That indirection was the source of "I edited the script and nothing changed":
the edit landed in week_sizedist.png while zonal90d_sizedist.png stayed stale.

  1) dashboard      -> {TAG}_dashboard.png   burdens, radiative feedback, size, gases, budget
  2) filmstrip      -> {TAG}_filmstrip.png   dT_rad + aerosol mass, COLUMN
                                             integrals when the run recorded them
                                             (frames_col_*), else the probe level
 2b) cross-section  -> {TAG}_crosssection.png zonal-mean lat-height sections of the
                                             same two fields -- the vertical
                                             transport the maps cannot show.
                                             Needs frames_zm_* (advection-MIP)
  3) size-dist      -> {TAG}_sizedist.png    dN/dlogDp evolution, two panels
                                             (global mean | 15S-15N), fixed axes
  4) drainage       -> {TAG}_drain.png      how fast the band empties and through
                                             where -- decay + e-folding time, the
                                             settling/advective split, and that
                                             split by latitude and by size. Drawn
                                             only for runs carrying the resolved
                                             D_* drain counters; it is the figure
                                             of the AER_SRC=fixed MICRO=off
                                             advection-only comparison.

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


def cfg(key, default=float("nan")):
    """One field of the run's own scenario stamp, by name.

    Read from the stamp rather than re-derived from the arrays because the stamp
    is what the run REFUSES to resume across (coupling.py's INJ_CFG), so it is
    the one description of the run that cannot silently disagree with it. Both
    arrays are append-only, so a key absent from an older file is a missing
    field, not an error.
    """
    k = [str(x) for x in ts["inj_cfg_keys"]] if "inj_cfg_keys" in ts.files else []
    return float(ts["inj_cfg"][k.index(key)]) if key in k else default


def phys(key, default=float("nan")):
    """One field of the run's PHYS_CFG stamp, by name (see cfg() above)."""
    k = [str(x) for x in ts["phys_cfg_keys"]] if "phys_cfg_keys" in ts.files else []
    return float(ts["phys_cfg"][k.index(key)]) if key in k else default


# ---- did this run have radiation and microphysics at all? -------------------
# Asked of the OUTPUT, not of an env var or a branch name: a run with RAD=0 never
# called the radiation driver, so aod550 is NaN at every step and dT_rad is
# identically zero. Drawing those panels anyway produces a blank map row and an
# empty axis -- figures that look like a result of zero rather than the absence
# of a calculation. The alternative, a MIP-only copy of this script, is the
# helper-script debt CLAUDE.md exists to stop; one data-driven test covers both
# the advection comparison and any future RAD=0 run.
_NO_INJ = cfg("INJ_SO2_TG_YR", 0.0) == 0 and cfg("INJ_H2SO4_TG_YR", 0.0) == 0
RAD_OFF = (("aod550" not in ts.files or not np.isfinite(ts["aod550"]).any())
           and ("dT_max" not in ts.files or not np.any(ts["dT_max"] != 0)))
MICRO_OFF = phys("MICRO_OFF", 0.0) == 1.0

# AER_SRC is stamped as an index into coupling.py's ('mam4','carma','fixed');
# 2 == the prescribed uniform PSD of the advection-only comparison, which is the
# only IC whose per-bin split at one level is also the global one (FIG 4d).
_IS_FIXED = cfg("AER_SRC") == 2
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
    # The title names the mechanism, so it has to follow the run: with INJ_* at
    # zero there is no injection and mass only falls, and "injection-driven mass
    # gain" over a monotonically decreasing curve is a caption contradicting the
    # data underneath it.
    ax.set_title(f"({letter}) burden: "
                 + ("number and mass drain out of the band" if _NO_INJ
                    else "number turnover vs injection-driven mass gain"))
    ax.set_xlabel("day")
    # M/M0 is on the twin axis, so a plain ax.legend() left it out entirely -- merge
    # both axes' handles or the panel's second curve goes unnamed
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # Upper right: on a monotonic drain both curves run down to the lower-right
    # corner and the legend sat on top of their endpoints -- the part of the
    # panel the eye goes to for the final drained fraction.
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7, framealpha=0.85)
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

# The BAND-average moments (added 2026-08-13) when the run recorded them, else
# the probe-level ones. Which is plotted changes what the panel MEANS -- one
# level near the top of the band against the whole band -- so the choice is
# carried into the panel title rather than left to the reader to assume.
_HAS_DOM = "reff_dom_nm" in ts.files and np.isfinite(ts["reff_dom_nm"]).any()
_REFF_KEY = "reff_dom_nm" if _HAS_DOM else "reff_nm"
_HAS_REFF = _REFF_KEY in ts.files


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
        return (days, 2.0 * np.asarray(ts[_REFF_KEY], float),
                "wet effective diameter")
    if not FRAMES_OK:
        return None
    # frames_col_num (column integral, #/m2) in preference to frames_num (one
    # level, #/kg). D_eff is a RATIO of moments of the same array, so the change
    # of units cancels exactly and only the sampling changes -- from 51.7 hPa to
    # the whole band, which is what the burdens beside this panel are quoted over.
    _iscol = "frames_col_num" in fr.files
    num = fr["frames_col_num"] if _iscol else get_num()
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
    return (fh / 24.0, deff,
            "dry effective diameter" + (" (column)" if _iscol else " (probe)"))


DEFF = effective_diameter()


def p_size(ax, letter="c"):
    if DEFF is None:
        ax.text(0.5, 0.5, "no reff_nm in the timeseries and no\nframes_num to "
                "rebuild it from", transform=ax.transAxes, ha="center",
                va="center", fontsize=9)
    else:
        _x, _y, _lab = DEFF
        ax.plot(_x, _y, "o-", color=C["b"], ms=3.5, lw=1.8, label=_lab)
    # WHERE this is measured goes in the title. It used to be a SINGLE LEVEL on a
    # figure whose other panels are all full-slab, which read as a slab quantity
    # purely from its company; runs carrying reff_dom_nm now really are full-slab
    # and the title has to keep the two apart.
    ax.set_title(f"({letter}) effective diameter of the aerosol  "
                 + ("(band average, air-mass weighted)" if _HAS_DOM
                    else f"@ {PROBE_HPA_LBL}"), fontsize=10)
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
# Under MICRO=off the micro stage is not a small source, it is an ABSENT one --
# every entry is exactly 0.0 -- so it is dropped from the budget rather than
# drawn as a flat line at zero with "(source)" in the legend.
BIG = ([] if MICRO_OFF else [("B_micro", "micro (source)", C["num"])]) \
    + [("B_adv_np", "advect (sink)", C["a"])]
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
# The AOD/forcing panel is dropped outright when the run had no radiation --
# not left blank -- and the remaining three go in one row rather than into a 2x2
# with a hole in it. Panel LETTERS follow the layout actually drawn, so (b) is
# whatever is second on the page; a figure whose panels skip a letter reads as
# one that lost a panel in production.
if RAD_OFF:
    fig, axs = plt.subplots(1, 3, figsize=(17.25, 4.9))
    fig.suptitle(f"{N_DAYS_RUN:.0f}-day "
                 + ("transport-only" if MICRO_OFF else "coupled")
                 + f" run ({TAG}): "
                 + ("advection + settling, no radiation and no microphysics"
                    if MICRO_OFF else "microphysics without radiation"),
                 fontsize=14, fontweight="bold")
    p_burden(axs[0], "a"); p_size(axs[1], "b"); p_budget(axs[2], "c")
else:
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
#
# COLUMN, not the probe level, whenever the run recorded it (frames_col_*, added
# for the advection-MIP). A single level shows horizontal transport only: aerosol
# that sinks or is lofted out of 51.7 hPa simply vanishes from the map, which on a
# figure about TRANSPORT is the one artifact that matters. The column integral
# cannot lose it -- it can only move within the band. Runs predating the diagnostic
# fall back to the probe level, and the title says which one is drawn.
_HAS_COL = FRAMES_OK and "frames_col_mas" in fr.files
_need = ("frames_col_dT", "frames_col_mas") if _HAS_COL else ("frames_dT", "frames_mas")
_missing = [k for k in _need if k not in fr.files] if FRAMES_OK else ["*"]
if not FRAMES_OK:
    print(f"  filmstrip SKIPPED: {TAG} frames file is unreadable")
elif _missing:
    print(f"  filmstrip SKIPPED: {TAG} frames file has no {', '.join(_missing)}"
          f" (has {', '.join(sorted(fr.files))})")
else:
    probe = float(fr["probe_hpa"])
if not _missing:
    # COL_KGM2 = sum(dp)/g, the air mass of the whole band [kg/m2]. On FIXED
    # pressure levels it is a constant, so the column SUM and the air-mass-weighted
    # column MEAN are the same field in different units -- which is why coupling.py
    # stores only the integral, and why the two colorbars below are one axis with
    # two scales rather than two maps.
    COL_KGM2 = float(fr["col_kgm2"]) if "col_kgm2" in fr.files else float("nan")
    if _HAS_COL:
        dT = fr["frames_col_dT"]                      # (nf,nlat,nlon) [K], column mean
        mass = fr["frames_col_mas"].sum(axis=1) * 1e6   # kg/m2 -> mg/m2
        MLAB = "column SO4 [mg m$^{-2}$]"
        MLAB2 = "column-mean SO4 [$\\times 10^{-9}$ kg/kg]"
        M2 = 1e3 / COL_KGM2                # mg/m2 -> 1e-9 kg/kg (mean MR)
        DLAB = "column-mean dT_rad [K]"
        WHERE = f"column integral ({float(fr['plev_hpa'][0]):.0f}-" \
                f"{float(fr['plev_hpa'][-1]):.0f} hPa)" \
                if "plev_hpa" in fr.files else "column integral"
    else:
        dT = fr["frames_dT"]                          # (nf,nlat,nlon)
        mass = fr["frames_mas"].sum(axis=1) * 1e9     # total SO4 MR, 1e-9 kg/kg
        MLAB = "SO4 [$\\times 10^{-9}$ kg/kg]"
        MLAB2 = M2 = None
        DLAB = "dT_rad [K]"
        WHERE = f"{probe:.1f} hPa"
    assert dT.shape[0] == nf, f"frames_dT has {dT.shape[0]} frames, frame_hours {nf}"
    sel = np.unique(np.linspace(0, nf - 1, min(5, nf)).astype(int))   # up to 5 snapshots

    proj = ccrs.PlateCarree()
    # PlateCarree axes are aspect-locked 2:1, so the figure height has to follow the
    # column width or the two rows float apart with a band of dead space between them
    COL_W = 4.2
    nrow = 1 if RAD_OFF else 2
    fig = plt.figure(figsize=(COL_W * len(sel), COL_W * nrow / 2 + 1.1),
                     constrained_layout=True)
    # The "and the means as well": on FIXED pressure levels the air-mass-weighted
    # column MEAN mixing ratio is the column SUM divided by the constant
    # COL_KGM2, so the mean map IS this map -- drawing it separately would be the
    # identical picture twice. It goes on the title as the one number that
    # converts the colorbar, because neither a second colorbar axis nor a
    # second label line fits beside a cartopy row at this figure width.
    fig.suptitle(f"Spatial evolution, {WHERE}: "
                 + ("aerosol mass" if RAD_OFF
                    else "radiative heating (top) & aerosol mass (bottom)")
                 + (f"\ncolumn mean = {M2:.3g} $\\times$ colorbar, in {MLAB2}"
                    if M2 is not None and np.isfinite(M2) else ""),
                 fontsize=13, fontweight="bold")
    # dT is strongly one-sided: ~-0.12 K of cooling against ~+3.9 K of heating.
    # Scale SYMMETRICALLY about 0 off the heating limb -- white stays at zero and
    # the cooling shows at its true (tiny) relative amplitude. Giving the negative
    # limb its own limits instead would stretch a 0.12 K range over half the
    # colormap and make trivial cooling look as strong as the heating.
    dpos = max(float(np.nanpercentile(dT[sel], 99.5)), 1e-6)
    dnorm = mcolors.Normalize(vmin=-dpos, vmax=dpos)
    # SO4 scale. Zero-anchored is right for an INJECTION run, where the plume is
    # a local enhancement over ~nothing and an offset floor would hide how much of
    # the map the plume has not reached. It is exactly wrong for a DRAINING run
    # started from a uniform background: the field spans 2.0 -> 1.4e-9 kg/kg and
    # anchoring at 0 spends 70% of the colormap on values that never occur, which
    # is why the drain filmstrip came out five near-identical yellow panels.
    # Decide from the data rather than from the run type: keep the zero anchor
    # only when the low end really does approach zero.
    mvmax = float(np.nanpercentile(mass[sel], 99.5))
    mvmin = float(np.nanpercentile(mass[sel], 0.5))
    if mvmin > 0.15 * mvmax:
        mnorm = mcolors.Normalize(mvmin, mvmax)
    else:
        mnorm = mcolors.Normalize(0, mvmax)
    # keep the two rows in their own lists -- add_subplot appends to fig.axes in
    # call order (axt0, axm0, axt1, ...), so slicing fig.axes interleaves the rows
    # and anchors each colorbar to a full-height mix of both
    top_axes, bot_axes = [], []
    for j, fi in enumerate(sel):
        if not RAD_OFF:
            axt = fig.add_subplot(nrow, len(sel), j + 1, projection=proj)
            axt.pcolormesh(lon, lat, dT[fi], cmap="RdBu_r",
                           norm=dnorm, transform=proj, shading="auto")
            axt.coastlines(linewidth=0.3, color="0.3")
            axt.set_title(f"day {fh[fi]/24:.1f}", fontsize=10)
            top_axes.append(axt)
        # the day label belongs to whichever row is on top
        axm = fig.add_subplot(nrow, len(sel), (nrow - 1) * len(sel) + j + 1,
                              projection=proj)
        axm.pcolormesh(lon, lat, mass[fi], cmap="viridis", norm=mnorm,
                       transform=proj, shading="auto")
        axm.coastlines(linewidth=0.3, color="w")
        if RAD_OFF:
            axm.set_title(f"day {fh[fi]/24:.1f}", fontsize=10)
        bot_axes.append(axm)
    if top_axes:
        sm1 = plt.cm.ScalarMappable(cmap="RdBu_r", norm=dnorm)
        fig.colorbar(sm1, ax=top_axes, orientation="vertical", fraction=0.02,
                     pad=0.01, aspect=18, label=DLAB, extend="max")
    sm2 = plt.cm.ScalarMappable(cmap="viridis", norm=mnorm)
    # the mass colorbar is the short one (aspect scaled down by nrow), so its
    # rotated label runs off the bar vertically at the default size, not
    # horizontally off the figure -- keep the text short and small
    cb2 = fig.colorbar(sm2, ax=bot_axes, orientation="vertical", fraction=0.02,
                       pad=0.01, aspect=18 // nrow,
                       extend="both" if mnorm.vmin > 0 else "max")
    cb2.set_label(MLAB, fontsize=9)
    save(fig, "filmstrip", bbox_inches="tight")

# =====================================================================
# FIG 2b -- ZONAL-MEAN CROSS-SECTIONS: lat-height, across the run
#   -> {TAG}_crosssection.png
# The filmstrip above answers "where has it spread"; this answers "how is it
# moving vertically" -- ascent in the tropical pipe, poleward-and-down along the
# Brewer-Dobson branches, and the settling tail off the bottom of the band, none
# of which a horizontal field can show at all. Needs frames_zm_* (advection-MIP
# addition); older runs simply do not get the figure.
# =====================================================================
_zm_need = ("frames_zm_mas", "plev_hpa")
_zm_missing = [k for k in _zm_need if k not in fr.files] if FRAMES_OK else ["*"]
if _zm_missing:
    print(f"  cross-section SKIPPED: {TAG} frames file has no "
          f"{', '.join(_zm_missing)} (predates the diagnostic)")
else:
    plev = np.asarray(fr["plev_hpa"], float)          # (nlev,) hPa, top first
    latz = lat_full()
    # (nf,NBINS,nlev,nlat) since 2026-08-14, (nf,nlev,nlat) before it. Sum the
    # bin axis by RANK rather than by run age, so both vintages plot.
    zmass = np.asarray(fr["frames_zm_mas"], float)
    if zmass.ndim == 4:
        zmass = zmass.sum(1)
    zmass = zmass * 1e9                                    # (nf,nlev,nlat)
    zdT = np.asarray(fr["frames_zm_dT"], float) if "frames_zm_dT" in fr.files \
        else np.zeros_like(zmass)
    sel_z = np.unique(np.linspace(0, nf - 1, min(5, nf)).astype(int))
    _rad = (not RAD_OFF) and np.any(zdT != 0)
    nrow_z = 2 if _rad else 1

    # LOG pressure axis, inverted. A 1-150 hPa band is 2+ decades and a linear
    # axis spends three quarters of its height on the bottom decade, squashing the
    # 1-10 hPa levels -- where the injected plume actually sits -- into a strip.
    fig, axs = plt.subplots(nrow_z, len(sel_z), squeeze=False,
                            figsize=(3.1 * len(sel_z), 2.9 * nrow_z + 0.9),
                            sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle("Zonal-mean cross-sections: "
                 + ("radiative heating (top) & aerosol mass (bottom)" if _rad
                    else "aerosol mass mixing ratio"),
                 fontsize=13, fontweight="bold")
    # Same norm rule as the filmstrip: symmetric about zero for dT (white == no
    # heating), zero-anchored for mass unless the field never approaches zero.
    zhi = float(np.nanpercentile(zmass[sel_z], 99.5))
    zlo = float(np.nanpercentile(zmass[sel_z], 0.5))
    znorm = mcolors.Normalize(zlo if zlo > 0.15 * zhi else 0.0, zhi)
    dzpos = max(float(np.nanpercentile(np.abs(zdT[sel_z]), 99.5)), 1e-6)
    dznorm = mcolors.Normalize(-dzpos, dzpos)
    for j, fi in enumerate(sel_z):
        if _rad:
            axs[0, j].pcolormesh(latz, plev, zdT[fi], cmap="RdBu_r",
                                 norm=dznorm, shading="auto")
            axs[0, j].set_title(f"day {fh[fi]/24:.1f}", fontsize=10)
        axm = axs[nrow_z - 1, j]
        axm.pcolormesh(latz, plev, zmass[fi], cmap="viridis", norm=znorm,
                       shading="auto")
        if not _rad:
            axm.set_title(f"day {fh[fi]/24:.1f}", fontsize=10)
        axm.set_xlabel("latitude [deg]")
    for ax in axs.ravel():
        ax.set_yscale("log")
        ax.set_ylim(plev.max(), plev.min())        # pressure increases downward
        ax.set_xlim(-90, 90)
        ax.set_xticks([-60, -30, 0, 30, 60])
        ax.grid(alpha=0.2, lw=0.4)
    for r in range(nrow_z):
        axs[r, 0].set_ylabel("pressure [hPa]")
    if _rad:
        fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=dznorm),
                     ax=list(axs[0]), fraction=0.02, pad=0.01, aspect=18,
                     label="zonal-mean dT_rad [K]", extend="both")
    fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=znorm),
                 ax=list(axs[nrow_z - 1]), fraction=0.02, pad=0.01, aspect=18,
                 label="zonal-mean SO4 [$\\times 10^{-9}$ kg/kg]",
                 extend="both" if znorm.vmin > 0 else "max")
    save(fig, "crosssection", bbox_inches="tight")

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
SIZEDIST_YLIM = (1e-2, 1e3)        # dN/dlogDp [cm-3 STP]

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
# Prefer the vertically-averaged PSD (frames_psd_num, added 2026-08-13) over the
# probe-level frames. frames_num is ONE LEVEL, so every distribution drawn from
# it describes 51.7 hPa while the burdens beside it are quoted over the whole
# 1-150 hPa band -- and in a drain run the probe level is the slowest-changing
# part of the band, so it understates exactly what the figure is for.
_HAS_PSD = FRAMES_OK and "frames_psd_num" in fr.files
_no_num = not FRAMES_OK or ("frames_num" not in fr.files and not _HAS_PSD)
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
        if _HAS_PSD:
            # (nf,40,nlat), already an air-mass-weighted vertical mean and a zonal
            # mean, so only the latitude reduction is left to do here
            psd = np.asarray(fr["frames_psd_num"], float)
            num_m = ((psd * wlat[None, None, :]).sum(axis=2)
                     / max(wlat.sum(), 1e-300))         # (nf,40), #/kg
        else:
            w = wlat[None, None, :, None]
            # get_num(), not fr["frames_num"]: this function is called once per
            # panel and an NpzFile member re-reads (1.6 GB here) on every access
            num = get_num()
            num_m = ((num * w).sum(axis=(2, 3))
                     / (w.sum() * num.shape[3]))        # (nf,40) regional mean, #/kg
        print(f"  sizedist: STP normalization, rho={RHO_STP:.5f} kg/m3 "
              + ("(band average over all levels)" if _HAS_PSD else
                 f"(probe level is {float(fr['probe_hpa']):.1f} hPa)")
              + f"; plotted {int(keep.sum())}/{keep.size} bins, "
              f"Dp >= {DP_MIN_NM:.0f} nm"
              + ("" if latmask is None else
                 f", {int(latmask.sum())}/{latmask.size} lat rows"), flush=True)
        return num_m * (RHO_STP / 1.0e6) / dlogDp[None, :]    # cm-3 STP per dlog10(Dp)

    # tropics: the injection band plus the ascending branch of the Brewer-Dobson
    # circulation, i.e. where the plume actually is for the first weeks
    # The tropics panel exists to show the ENHANCEMENT over the global mean in the
    # injection band -- read off the offset between two identically-scaled panels.
    # With INJ_* at zero there is no plume and no enhancement to read: the two
    # panels are the same distribution twice, and the second one invites a
    # comparison that has nothing in it.
    TROPIC_LAT = 15.0
    PANELS = [("global", dNdlogDp_of())]
    if not _NO_INJ:
        PANELS = [("(a) global", PANELS[0][1]),
                  (f"(b) tropics ({TROPIC_LAT:.0f}S-{TROPIC_LAT:.0f}N)",
                   dNdlogDp_of(latmask=np.abs(lat_full()) <= TROPIC_LAT))]

    cmap = plt.cm.viridis
    # sharey (not just equal set_ylim) so the right panel loses its tick labels too:
    # repeating them invites reading the panels as separately scaled
    fig, axs = plt.subplots(1, len(PANELS), squeeze=False,
                            figsize=(7.6 if len(PANELS) == 1 else 13.5, 5.8),
                            sharex=True, sharey=True)
    axs = axs[0]
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
                 + ("over the whole band (air-mass-weighted vertical mean)"
                    if _HAS_PSD else f"@ {float(fr['probe_hpa']):.1f} hPa"),
                 fontsize=12, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, fh[-1] / 24))
    fig.colorbar(sm, ax=axs, label="day", fraction=0.03, pad=0.11)
    # no tight_layout: it fights the colorbar's shared-axes placement. right= has to
    # leave room for BOTH the bar and its pad, or the bar is pushed off the canvas.
    fig.subplots_adjust(left=0.07 if len(PANELS) > 1 else 0.11,
                        right=0.855 if len(PANELS) > 1 else 0.80,
                        top=0.86, bottom=0.11, wspace=0.06)
    save(fig, "sizedist")

# =====================================================================
# FIG 4 -- drainage: how fast the band empties, and through where
#   -> {TAG}_drain.png, only for runs that recorded the resolved drain
#      counters (D_* in the timeseries, added 2026-08-13).
# =====================================================================
# This is THE figure of the advection-only inter-model comparison
# (AER_SRC=fixed MICRO=off RAD=0): with no source and no microphysics the
# burden only falls, and the question is how fast, through which face, at which
# latitudes and at which sizes. The other three figures answer none of that --
# the dashboard's budget panel has the two channels globally summed, and the
# filmstrip shows one level.
#
# The two channels are NEVER merged into one curve, because they are the two
# halves the comparison is meant to separate: settling depends on the model's
# temperature and nothing else and should be nearly model-independent, while the
# advective flux through the same face IS the residual circulation and is where
# two dycores are expected to disagree.
_DK = ('D_setM_lat', 'D_vfM_lat', 'D_setM_bin', 'D_vfM_bin')
_have_drain = all(k in ts.files for k in _DK)
if _have_drain:
    # Cumulative, burden units, positive = left the band through the bottom.
    dlat_s = np.asarray(ts["D_setM_lat"], float)      # (nt, nlat)
    dlat_v = np.asarray(ts["D_vfM_lat"], float)
    dbin_s = np.asarray(ts["D_setM_bin"], float)      # (nt, nbins)
    dbin_v = np.asarray(ts["D_vfM_bin"], float)
    # A RESUME across the commit that added these keys NaN-pads the earlier
    # records (see coupling.py's _TS_VEC note), so take the last row that is
    # actually finite rather than [-1] and say so if they differ.
    _fin = np.where(np.isfinite(dlat_s).all(1) & np.isfinite(dlat_v).all(1))[0]
    _have_drain = _fin.size > 0
if not _have_drain:
    print(f"  drainage figure SKIPPED: {TAG} has no finite D_* drain counters "
          f"(pre-2026-08-13 run, or resumed onto one)", flush=True)
else:
    iL = int(_fin[-1])
    # A smoke run is a fraction of a day, and "%.0f days" prints that as "0" --
    # a figure titled "0 days" for a run that clearly integrated something.
    def _dfmt(d):
        return f"{d:.1f}" if d < 10 else f"{d:.0f}"
    if iL != len(days) - 1:
        print(f"  drainage: last finite drain record is day {days[iL]:.1f} of "
              f"{days[-1]:.1f} (earlier records NaN-padded by a RESUME)", flush=True)
    latd = (np.asarray(fr["lat"]) if (fr is not None and "lat" in fr.files)
            else np.linspace(-90, 90, dlat_s.shape[1]))
    # everything below is a FRACTION OF THE INITIAL BURDEN -- the run is linear in
    # the aerosol under MICRO=off, so the absolute scale carries no information
    fs, fv = dlat_s / M0, dlat_v / M0
    tot_s, tot_v = fs.sum(1), fv.sum(1)               # (nt,) global cumulative

    def p_decay(ax, letter="a"):
        """(a) what is left, and the e-folding time that implies."""
        mrat = ts["Mburden"] / M0
        ax.plot(days, mrat, "-", color=C["mass"], lw=2.0, label="M / M0  (mass)")
        ax.plot(days, ts["Nburden"] / N0, "-", color=C["num"], lw=1.2, alpha=0.8,
                label="N / N0  (number)")
        ax.axhline(1 / np.e, color="0.5", ls=":", lw=1.0)
        ax.text(days[1] if len(days) > 1 else 0, 1 / np.e, " 1/e", va="bottom",
                ha="left", fontsize=8, color="0.4")
        # Two residence times, and they are different numbers whenever the decay
        # is not a single exponential (it is not: the fast-draining lowest levels
        # empty first, so the rate slows). Quote both rather than pick one.
        #   tau_fit  : slope of ln(M/M0) over the LAST HALF -- the late-time rate
        #   tau_bulk : t / ln(M0/M) at the end -- the whole-run average
        good = np.isfinite(mrat) & (mrat > 0)
        half = good & (days >= 0.5 * days[-1])
        txt = []
        if half.sum() >= 3:
            sl = np.polyfit(days[half], np.log(mrat[half]), 1)[0]
            if sl < 0:
                txt.append(f"tau (late, day {_dfmt(days[half][0])}+) = {-1/sl:6.1f} d")
        if mrat[-1] > 0 and mrat[-1] < 1:
            txt.append(f"tau (run mean)          = "
                       f"{days[-1] / np.log(1 / mrat[-1]):6.1f} d")
        txt.append(f"drained by day {_dfmt(days[-1])}      = {1 - mrat[-1]:6.1%}")
        ax.text(0.97, 0.95, "\n".join(txt), transform=ax.transAxes, va="top",
                ha="right", fontsize=8.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="w", ec="0.8", alpha=0.9))
        ax.set_yscale("log")
        ax.set_title(f"({letter}) band burden -- a pure drain, no source")
        ax.set_xlabel("day"); ax.set_ylabel("remaining / initial")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
        ax.grid(alpha=0.25, which="both")

    def p_channel(ax, letter="b"):
        """(b) the two channels over time, cumulative, plus the closure check."""
        ax.plot(days, 100 * tot_v, "-", color=C["a"], lw=2.0,
                label="advective flux through the base")
        ax.plot(days, 100 * tot_s, "-", color=C["b"], lw=2.0,
                label="gravitational settling")
        ax.plot(days, 100 * (tot_s + tot_v), "--", color="0.35", lw=1.2,
                label="total drained")
        # The band has FOUR loss channels, not two. Plotting the base pair against
        # 1-M/M0 and calling the gap a "scheme residual" was wrong: it was
        # dominated by the two channels left off the figure -- outflow through the
        # TOP face and the polar-cap stirring -- so it charged real physics to the
        # advection scheme. Both are small here (0.42% and 0.30% of M0 over 90
        # days against 45.6% through the base) but they are not zero, and the
        # figure's own closure check is what has to prove that rather than assume
        # it. B_vf_in/B_adv_pol are stored as SIGNED and negative for a loss;
        # negate so every curve on this panel is a positive amount lost.
        if "B_vf_in" in ts.files:
            ax.plot(days, -100 * np.asarray(ts["B_vf_in"], float), "-",
                    color=C["so2"], lw=1.2, label="outflow through the top face")
        if "B_adv_pol" in ts.files:
            ax.plot(days, -100 * np.asarray(ts["B_adv_pol"], float), "-",
                    color=C["h2so4"], lw=1.2, label="polar-cap stirring")
        lost = 100 * (1 - ts["Mburden"] / M0)
        ax.plot(days, lost, ":", color=C["mass"], lw=1.6, label="1 - M/M0")
        # The residual the RUN reports, in the run's own terms: what the advection
        # stages took out of the slab minus what actually crossed its faces. That
        # is the number the step log prints as "advection numerical residual", so
        # the figure and the log cannot drift apart.
        _res = (float(np.asarray(ts["B_adv_np"])[-1]
                      + np.asarray(ts["B_adv_pol"])[-1]
                      - np.asarray(ts["B_vf_in"])[-1]
                      - np.asarray(ts["B_vf_out"])[-1])
                if all(k in ts.files for k in
                       ("B_adv_np", "B_adv_pol", "B_vf_in", "B_vf_out")) else float("nan"))
        ax.text(0.03, 0.95, f"advection numerical residual at day "
                f"{_dfmt(days[-1])}: {100 * _res:+.4f} % of M0",
                transform=ax.transAxes, va="top",
                ha="left", fontsize=8.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="w", ec="0.8", alpha=0.9))
        ax.set_title(f"({letter}) cumulative loss by channel (all four)")
        ax.set_xlabel("day"); ax.set_ylabel("% of initial mass")
        add_headroom(ax, 0.18)
        ax.legend(loc="center left", fontsize=8, framealpha=0.85)
        ax.grid(alpha=0.25)

    def p_lat(ax, letter="c"):
        """(c) WHERE it leaves. Cumulative loss per latitude row, both channels."""
        ax.plot(latd, 100 * fv[iL], "-", color=C["a"], lw=1.6, label="advective")
        ax.plot(latd, 100 * fs[iL], "-", color=C["b"], lw=1.6, label="settling")
        ax.axhline(0, color="0.6", lw=0.8)
        # NET, not gross: each row is summed over longitude and over every
        # substep, so a row where air ascends into the band cancels one where it
        # descends. A negative advective value is therefore a net INFLOW of
        # aerosol-free air, i.e. dilution -- not a sign error.
        ax.fill_between(latd, 0, 100 * fv[iL], where=(fv[iL] < 0),
                        color=C["a"], alpha=0.15)
        ax.set_title(f"({letter}) net loss through the base by latitude "
                     f"(day {_dfmt(days[iL])})")
        ax.set_xlabel("latitude [deg]")
        ax.set_ylabel("% of initial mass, per lat row")
        ax.set_xlim(-90, 90); ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax.legend(loc="upper center", fontsize=8, framealpha=0.85)
        ax.grid(alpha=0.25)

    def p_size(ax, letter="d"):
        """(d) drainage-vs-size, the curve MICRO=off gets for free.

        The bins are independent passive tracers, so one run is a 40-point sweep
        over fall speed. Normalizing each bin by ITS OWN initial mass is what
        makes the panel a rate curve rather than a picture of the initial PSD --
        and that per-bin initial mass is only recoverable because the fixed-PSD
        IC is uniform in space, so the probe level's bin split IS the global one.
        Without frames, fall back to the raw per-bin loss.
        """
        f0 = None
        if FRAMES_OK and "frames_mas" in fr.files and _IS_FIXED:
            w = wlat_full()[None, :, None]
            m0b = (fr["frames_mas"][0] * w).sum(axis=(1, 2))       # (nbins,)
            if m0b.sum() > 0:
                f0 = m0b / m0b.sum()
        x = Dp_mid if Dp_mid is not None else np.arange(dbin_s.shape[1]) + 1.0
        if f0 is not None:
            den = np.where(f0 > 0, f0 * M0, np.nan)
            ys, yv = dbin_s[iL] / den, dbin_v[iL] / den
            ylab = "% of that bin's initial mass"
        else:
            ys, yv = dbin_s[iL] / M0, dbin_v[iL] / M0
            ylab = "% of TOTAL initial mass"
        ax.plot(x, 100 * yv, "o-", color=C["a"], ms=3, lw=1.4, label="advective")
        ax.plot(x, 100 * ys, "o-", color=C["b"], ms=3, lw=1.4, label="settling")
        ax.plot(x, 100 * (ys + yv), "--", color="0.35", lw=1.2, label="total")
        if Dp_mid is not None:
            ax.set_xscale("log"); ax.set_xlabel("dry diameter Dp [nm]")
        else:
            ax.set_xlabel("size bin")
        ax.set_title(f"({letter}) loss by size (day {_dfmt(days[iL])})")
        ax.set_ylabel(ylab)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
        ax.grid(alpha=0.25, which="both")

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 9))
    fig.suptitle(f"Drainage out of the band ({TAG}): "
                 f"{'transport-only' if _IS_FIXED else 'aerosol'} run, "
                 f"{_dfmt(days[iL])} days", fontsize=14, fontweight="bold")
    p_decay(axs[0, 0], "a"); p_channel(axs[0, 1], "b")
    p_lat(axs[1, 0], "c"); p_size(axs[1, 1], "d")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "drain")

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
if RAD_OFF:
    print("  radiation: OFF (no dT_rad, no ARF, no AOD)", flush=True)
else:
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
