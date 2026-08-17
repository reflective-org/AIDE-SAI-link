#!/usr/bin/env python3
"""Animated versions of the filmstrip panels (plot_run.py FIG 2 and 2b).

Writes one GIF per field, using the SAME colormaps and symmetric/zero-anchored
norms as the filmstrip so a frame of the GIF and a column of the PNG are directly
comparable:

  {TAG}_dTrad.gif      radiative heating anomaly dT_rad [K]      (RdBu_r)
  {TAG}_so4.gif        total SO4 mass                            (viridis)
  {TAG}_so4_zm.gif     SO4 zonal-mean lat-height cross-section   (viridis)
  {TAG}_dTrad_zm.gif   dT_rad zonal-mean cross-section           (RdBu_r)

The two maps are COLUMN integrals when the run recorded frames_col_* and the
probe level otherwise; the two cross-sections need frames_zm_* and are skipped
for runs that predate it. Which one is drawn is in each GIF's own title.

  python3 gif_run.py [TAG] [--fps N] [--width PX] [--stride N]
                     [--log [--decades N]] [--massdens]   # TAG: zonal90d

Frames are stamped with the model date (365-day calendar from 1996-01-01, or
from H0 if the run set it), not a day index. On a multi-year run reach for
--log --massdens: the plume decays by three decades, and the cross-sections
otherwise animate a mixing ratio, which makes a draining plume look like it is
filling the upper stratosphere. --stride keeps a long run's GIF scrubbable.

Frames are rendered to a temp dir and assembled by ffmpeg with a per-GIF
palette (palettegen/paletteuse). Straight PIL GIF writing quantizes each frame
independently, which makes the colorbar and the smooth aerosol plume shimmer
between frames; one global palette holds the colors still.
"""
import os
import sys
import glob
import shutil
import argparse
import subprocess
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

p = argparse.ArgumentParser()
p.add_argument("tag", nargs="?", default="zonal90d")
p.add_argument("--fps", type=float, default=8.0, help="frames per second")
p.add_argument("--width", type=int, default=900, help="output width in pixels")
p.add_argument("--hold", type=float, default=1.2,
               help="seconds to hold the final frame before the loop restarts")
p.add_argument("--stride", type=int, default=0,
               help="use every Nth frame (0 = auto: cap the GIF at ~400 frames)")
p.add_argument("--log", action="store_true",
               help="log colour scale for the SO4 fields. Use for runs whose "
                    "plume decays by more than a decade; a linear scale goes "
                    "black after the first year")
p.add_argument("--decades", type=float, default=4.0,
               help="span of the --log ramp (default 4)")
p.add_argument("--massdens", action="store_true",
               help="cross-sections animate tracer MASS per unit altitude "
                    "(q*p*cos lat), normalised to the initial peak, instead of "
                    "the raw mixing ratio. The y axis is log-pressure, i.e. "
                    "linear in altitude, so equal areas then hold equal mass. "
                    "Without it a decaying plume LOOKS like it fills the upper "
                    "stratosphere: q stays large where the air density, and so "
                    "the mass per km, has fallen by three decades")
args = p.parse_args()
TAG = args.tag

# The CESM h1 record starts 1996-01-01 00Z and the model calendar has 365-day
# years, so an hour index IS a date. H0 shifts the start, read exactly as
# coupling.py reads it. Frames are stamped with the date rather than "day 3285":
# on a multi-year run a day index is unreadable, and on a 90-day run the date
# still resolves to the day.
H0 = int(os.environ.get("H0", "0"))
_MLEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_MNAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def stamp(hours):
    """Frame hour -> '1998 Apr 11' on the model's 365-day calendar."""
    y, doy = divmod((H0 + hours) / 24.0, 365.0)
    m = 0
    while m < 11 and doy >= _MLEN[m]:
        doy -= _MLEN[m]
        m += 1
    return f"{1996 + int(y)} {_MNAME[m]} {int(doy) + 1:2d}"

if not os.path.exists(f"coupled_frames_{TAG}.npz"):
    have = sorted(g[len("coupled_frames_"):-len(".npz")]
                  for g in glob.glob("coupled_frames_*.npz")
                  if not g.endswith("_ckpt.npz"))
    sys.exit(f"no run named '{TAG}'.\navailable tags:\n  " + "\n  ".join(have))
if shutil.which("ffmpeg") is None:
    sys.exit("ffmpeg not found on PATH -- needed to assemble the GIFs")

fr = np.load(f"coupled_frames_{TAG}.npz")

nlat, nlon = 192, 288
lat = np.asarray(fr["lat"]) if "lat" in fr.files else np.linspace(-90, 90, nlat)
lon = np.asarray(fr["lon"]) if "lon" in fr.files else \
    np.linspace(0, 360, nlon, endpoint=False)
probe = float(fr["probe_hpa"])
fh = np.asarray(fr["frame_hours"], float)
# Subsample BEFORE anything is read into memory or a colour scale is taken off
# it. A 15-year run at 5-day frames is 1096 frames: 2+ minutes of animation and
# a GIF nobody can scrub. The cap is on frame COUNT, so short runs are untouched.
STRIDE = args.stride if args.stride > 0 else max(1, int(np.ceil(len(fh) / 400.0)))
_SEL = slice(None, None, STRIDE)
fh = fh[_SEL]
if STRIDE > 1:
    print(f"  {TAG}: every {STRIDE}th frame ({len(fh)} of "
          f"{len(np.asarray(fr['frame_hours']))}) -- pass --stride 1 for all",
          flush=True)
# COLUMN integrals when the run recorded them, for the same reason plot_run.py's
# filmstrip prefers them: at one level, aerosol that sinks or is lofted out of
# 51.7 hPa disappears from the animation, and on a movie ABOUT transport that
# reads as loss. The column integral can only redistribute within the band.
_HAS_COL = "frames_col_mas" in fr.files
if _HAS_COL:
    dT = fr["frames_col_dT"][_SEL]              # (nf,nlat,nlon) [K] column mean
    mass = fr["frames_col_mas"][_SEL].sum(axis=1) * 1e6      # kg/m2 -> mg/m2
    MLAB = "column SO4 burden [mg m$^{-2}$]"
    DLAB = "column-mean dT_rad [K]"
    WHERE = "column"
else:
    dT = fr["frames_dT"][_SEL]                  # (nf,nlat,nlon)
    mass = fr["frames_mas"][_SEL].sum(axis=1) * 1e9   # (nf,nlat,nlon), 1e-9 kg/kg
    MLAB = "SO4 [x1e-9 kg/kg]"
    DLAB = "dT_rad [K]"
    WHERE = f"{probe:.1f} hPa"
nf = dT.shape[0]

# Same scaling rule as the filmstrip, but taken over ALL frames rather than the
# 5 sampled columns: dT symmetric about 0 off the heating limb (white == zero,
# cooling shows at its true tiny amplitude), mass zero-anchored.
dpos = max(float(np.nanpercentile(dT, 99.8)), 1e-6)
dnorm = mcolors.Normalize(vmin=-dpos, vmax=dpos)
# Same zero-anchor-only-if-it-earns-it rule as the filmstrip (plot_run.py FIG 2),
# because the module docstring promises the GIF and the PNG share their norms --
# changing one and not the other silently breaks the comparison it advertises.
_mhi = float(np.nanpercentile(mass, 99.8))
_mlo = float(np.nanpercentile(mass, 0.2))
DECADES = args.decades   # span of the --log ramp, as in pulse_crosssection.py


def mass_norm(hi):
    """Linear by default; log on --log, floored a fixed number of decades under
    the top so the span is stated by the figure rather than set by whatever the
    faintest surviving cell happens to be."""
    if args.log:
        return mcolors.LogNorm(vmin=hi / 10.0 ** DECADES, vmax=hi)
    return mcolors.Normalize(_mlo if _mlo > 0.15 * hi else 0.0, hi)


mnorm = mass_norm(_mhi)
_LOGSUF = ",  log scale" if args.log else ""   # the ticks state the range

proj = ccrs.PlateCarree()
FIELDS = [
    dict(name="dTrad", data=dT, cmap="RdBu_r", norm=dnorm, coast="0.3",
         label=DLAB, title="radiative heating anomaly"),
    dict(name="so4", data=mass, cmap="viridis", norm=mnorm, coast="w",
         label=MLAB + _LOGSUF, title="SO4 mass" + (" burden" if _HAS_COL
                                                   else " mixing ratio")),
]
# ---- zonal-mean cross-sections: the vertical companion to the maps ----------
# Same frames, same colormaps, but lat-height -- this is the one that shows the
# plume rising in the tropical pipe and draining out of the bottom of the band.
# Rendered by render_zm() below rather than render(): no map projection, no
# coastlines, and a LOG pressure axis (a 1-150 hPa band is 2+ decades, and a
# linear axis squashes the levels the plume is actually in into a strip).
ZFIELDS = []
if "frames_zm_mas" in fr.files and "plev_hpa" in fr.files:
    plev = np.asarray(fr["plev_hpa"], float)
    # bin-resolved (nf,NBINS,nlev,nlat) since 2026-08-14; sum by RANK so older
    # bin-summed frames files still animate (same test as plot_run.py FIG 2b)
    zmass = np.asarray(fr["frames_zm_mas"], float)[_SEL]
    if zmass.ndim == 4:
        zmass = zmass.sum(1)
    zmass = zmass * 1e9
    ZLAB = "zonal-mean SO4 [x1e-9 kg/kg]"
    if args.massdens:
        # dz = H dp/p, so dM/dz ~ q*p*cos(lat). Normalised to the initial peak,
        # which also makes the ramp anchor a stated number rather than a
        # percentile of whatever the run happened to contain.
        _cl = np.cos(np.deg2rad(np.asarray(lat, float))).clip(0.0)
        zmass = zmass * plev[None, :, None] * _cl[None, None, :]
        zmass = zmass / zmass[0].max()
        # Named in words, not symbols: the field is q*p*cos(lat), i.e. mass per
        # unit ALTITUDE (the q*p part, which divides out the model's uneven layer
        # thickness) per unit LATITUDE (the cos(lat) part, which weights each
        # zonal ring by the area it actually covers).
        ZLAB = "mass per km per degree latitude / the initial peak"
    zhi = 1.0 if args.massdens else float(np.nanpercentile(zmass, 99.8))
    zlo = float(np.nanpercentile(zmass, 0.2))
    ZFIELDS.append(dict(
        name="so4_zm", data=zmass, cmap="viridis",
        norm=(mcolors.LogNorm(vmin=zhi / 10.0 ** DECADES, vmax=zhi) if args.log
              else mcolors.Normalize(zlo if zlo > 0.15 * zhi else 0.0, zhi)),
        label=ZLAB + _LOGSUF,
        title="SO4 zonal-mean cross-section"))
    if "frames_zm_dT" in fr.files and np.any(np.asarray(fr["frames_zm_dT"]) != 0):
        zdT = np.asarray(fr["frames_zm_dT"], float)[_SEL]
        _zd = max(float(np.nanpercentile(np.abs(zdT), 99.8)), 1e-6)
        ZFIELDS.append(dict(
            name="dTrad_zm", data=zdT, cmap="RdBu_r",
            norm=mcolors.Normalize(-_zd, _zd),
            label="zonal-mean dT_rad [K]",
            title="radiative heating zonal-mean cross-section"))
else:
    print(f"  {TAG} frames file has no frames_zm_* "
          f"-- skipping the cross-section GIFs", flush=True)
# A RAD=0 run (the advection-only comparison is one) carries dT_rad as exactly
# zero at every frame, and the dTrad GIF is then a uniformly white map animated
# over 90 days -- several MB of nothing, and worse, a figure that LOOKS like a
# result. Drop the field rather than render it, and say which run it was.
if not np.any(dT != 0):
    FIELDS = [f for f in FIELDS if f["name"] != "dTrad"]
    print(f"  dT_rad is identically zero in {TAG} (radiation off) "
          f"-- skipping the dTrad GIF", flush=True)


def render(field, outdir):
    """Write one PNG per frame. Reuse a single figure and just swap the mesh's
    array -- rebuilding the axes 46x re-projects the coastlines every time and
    is what makes the naive version take minutes instead of seconds."""
    # PlateCarree is aspect-locked 2:1; leave room for the title and colorbar
    dpi = 100.0
    fig = plt.figure(figsize=(args.width / dpi, args.width / dpi * 0.60), dpi=dpi,
                     constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    mesh = ax.pcolormesh(lon, lat, field["data"][0], cmap=field["cmap"],
                         norm=field["norm"], transform=proj, shading="auto")
    ax.coastlines(linewidth=0.4, color=field["coast"])
    fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.024, pad=0.01,
                 aspect=22, label=field["label"],
                 extend="both" if field["norm"].vmin > 0 else "max")
    ttl = fig.suptitle("", fontsize=12, fontweight="bold")

    paths = []
    for i in range(nf):
        # set_array wants the flattened C-order values for shading="auto"
        mesh.set_array(field["data"][i].ravel())
        ttl.set_text(f"{field['title']}, {WHERE}   --   {stamp(fh[i])}")
        path = os.path.join(outdir, f"f{i:04d}.png")
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def render_zm(field, outdir):
    """render() for a lat-height section: no projection, log-p axis, no coasts."""
    dpi = 100.0
    fig = plt.figure(figsize=(args.width / dpi, args.width / dpi * 0.60), dpi=dpi,
                     constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    mesh = ax.pcolormesh(lat, plev, field["data"][0], cmap=field["cmap"],
                         norm=field["norm"], shading="auto")
    # on --log, anything at or below the floor is masked by LogNorm; paint it
    # neutral grey so "empty" reads as empty rather than as the palest data colour
    ax.set_facecolor("0.92")
    ax.set_yscale("log")
    ax.set_ylim(plev.max(), plev.min())          # pressure increases downward
    ax.set_xlim(-90, 90); ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xlabel("latitude [deg]"); ax.set_ylabel("pressure [hPa]")
    ax.grid(alpha=0.2, lw=0.4)
    fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.024, pad=0.01,
                 aspect=22, label=field["label"],
                 extend="both" if field["norm"].vmin > 0 else "max")
    ttl = fig.suptitle("", fontsize=12, fontweight="bold")

    paths = []
    for i in range(nf):
        mesh.set_array(field["data"][i].ravel())
        ttl.set_text(f"{field['title']}   --   {stamp(fh[i])}")
        path = os.path.join(outdir, f"f{i:04d}.png")
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def assemble(indir, out, fps, hold_s):
    """ffmpeg: build a global palette from all frames, then map every frame
    through it. bayer dithering keeps the smooth plume gradients from banding
    without the crawling speckle sierra2 gives on an animation."""
    pal = os.path.join(indir, "palette.png")
    src = ["-framerate", f"{fps:g}", "-i", os.path.join(indir, "f%04d.png")]
    subprocess.run(["ffmpeg", "-y", "-v", "error", *src,
                    "-vf", "palettegen=stats_mode=full", pal], check=True)
    # hold the last frame: repeat it enough times to cover hold_s at this fps
    extra = max(0, int(round(hold_s * fps)))
    last = os.path.join(indir, f"f{nf - 1:04d}.png")
    for k in range(extra):
        shutil.copyfile(last, os.path.join(indir, f"f{nf + k:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-v", "error", *src, "-i", pal,
                    "-lavfi", "paletteuse=dither=bayer:bayer_scale=3",
                    "-loop", "0", out], check=True)


for field in [(f, render) for f in FIELDS] + [(f, render_zm) for f in ZFIELDS]:
    field, _draw = field
    out = f"{TAG}_{field['name']}.gif"
    with tempfile.TemporaryDirectory(prefix=f"gif_{field['name']}_") as td:
        _draw(field, td)
        assemble(td, out, args.fps, args.hold)
    print(f"wrote {out}  ({nf} frames, {stamp(fh[0])} - {stamp(fh[-1])}, "
          f"{os.path.getsize(out)/1e6:.1f} MB)", flush=True)
