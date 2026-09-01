#!/usr/bin/env python3
"""Animated versions of the filmstrip panels (plot_run.py FIG 2).

Writes two GIFs at the probe level, one per field, using the SAME colormaps and
symmetric/zero-anchored norms as the filmstrip so a frame of the GIF and a
column of the PNG are directly comparable:

  {TAG}_dTrad.gif   radiative heating anomaly dT_rad [K]      (RdBu_r)
  {TAG}_so4.gif     total SO4 mass mixing ratio [1e-9 kg/kg]  (viridis)

Runs made with FRAME_LEVELS=all also get the LATITUDE-HEIGHT animations, which
are the ones that show the plume descending rather than just spreading:

  {TAG}_zonal_so4.gif    zonal-mean SO4, latitude x pressure   (viridis, log)
  {TAG}_zonal_dTrad.gif  zonal-mean dT_rad, latitude x pressure (RdBu_r)

They are skipped with a note under FRAME_LEVELS=probe, where only one level
exists and a lat-height panel cannot be built at all.

  python3 gif_run.py [TAG] [--fps N] [--width PX]     # TAG defaults to zonal90d

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
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

p = argparse.ArgumentParser()
p.add_argument("tag", nargs="?", default="zonal90d")
p.add_argument("--fps", type=float, default=8.0, help="frames per second")
p.add_argument("--width", type=int, default=900, help="output width in pixels")
p.add_argument("--hold", type=float, default=1.2,
               help="seconds to hold the final frame before the loop restarts")
args = p.parse_args()
TAG = args.tag

if not os.path.exists(f"coupled_frames_{TAG}.npz"):
    have = sorted(g[len("coupled_frames_"):-len(".npz")]
                  for g in glob.glob("coupled_frames_*.npz")
                  if not g.endswith("_ckpt.npz"))
    sys.exit(f"no run named '{TAG}'.\navailable tags:\n  " + "\n  ".join(have))
if shutil.which("ffmpeg") is None:
    sys.exit("ffmpeg not found on PATH -- needed to assemble the GIFs")

fr = np.load(f"coupled_frames_{TAG}.npz")

nlat, nlon = 192, 288
lat = np.linspace(-90, 90, nlat)
lon = np.linspace(0, 360, nlon, endpoint=False)
probe = float(fr["probe_hpa"])
fh = np.asarray(fr["frame_hours"], float)
dT = fr["frames_dT"]                      # (nf,nlat,nlon)
mass = fr["frames_mas"].sum(axis=1) * 1e9   # (nf,nlat,nlon), 1e-9 kg/kg
nf = dT.shape[0]

# Same scaling rule as the filmstrip, but taken over ALL frames rather than the
# 5 sampled columns: dT symmetric about 0 off the heating limb (white == zero,
# cooling shows at its true tiny amplitude), mass zero-anchored.
dpos = max(float(np.nanpercentile(dT, 99.8)), 1e-6)
dnorm = mcolors.Normalize(vmin=-dpos, vmax=dpos)
mnorm = mcolors.Normalize(0.0, float(np.nanpercentile(mass, 99.8)))

proj = ccrs.PlateCarree()
FIELDS = [
    dict(name="dTrad", data=dT, cmap="RdBu_r", norm=dnorm, coast="0.3",
         label="dT_rad [K]", title="radiative heating anomaly"),
    dict(name="so4", data=mass, cmap="viridis", norm=mnorm, coast="w",
         label="SO4 [x1e-9 kg/kg]", title="SO4 mass mixing ratio"),
]


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
                 aspect=22, label=field["label"], extend="max")
    ttl = fig.suptitle("", fontsize=12, fontweight="bold")

    paths = []
    for i in range(nf):
        # set_array wants the flattened C-order values for shading="auto"
        mesh.set_array(field["data"][i].ravel())
        ttl.set_text(f"{field['title']} @ {probe:.1f} hPa   --   day {fh[i]/24:5.1f}")
        path = os.path.join(outdir, f"f{i:04d}.png")
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def assemble(indir, out, fps, hold_s, n=None):
    """ffmpeg: build a global palette from all frames, then map every frame
    through it. bayer dithering keeps the smooth plume gradients from banding
    without the crawling speckle sierra2 gives on an animation."""
    pal = os.path.join(indir, "palette.png")
    src = ["-framerate", f"{fps:g}", "-i", os.path.join(indir, "f%04d.png")]
    subprocess.run(["ffmpeg", "-y", "-v", "error", *src,
                    "-vf", "palettegen=stats_mode=full", pal], check=True)
    # hold the last frame: repeat it enough times to cover hold_s at this fps
    extra = max(0, int(round(hold_s * fps)))
    # n defaults to the probe-frame count, but the zonal gifs can legitimately
    # have fewer frames (a resume onto a probe-only checkpoint starts their
    # history late), so the caller passes their own count.
    n = nf if n is None else n
    last = os.path.join(indir, f"f{n - 1:04d}.png")
    for k in range(extra):
        shutil.copyfile(last, os.path.join(indir, f"f{n + k:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-v", "error", *src, "-i", pal,
                    "-lavfi", "paletteuse=dither=bayer:bayer_scale=3",
                    "-loop", "0", out], check=True)


for field in FIELDS:
    out = f"{TAG}_{field['name']}.gif"
    with tempfile.TemporaryDirectory(prefix=f"gif_{field['name']}_") as td:
        render(field, td)
        assemble(td, out, args.fps, args.hold)
    print(f"wrote {out}  ({nf} frames, {fh[-1]/24:.0f} days, "
          f"{os.path.getsize(out)/1e6:.1f} MB)", flush=True)


# =====================================================================
# latitude-height animations -- FRAME_LEVELS=all only
# =====================================================================
# These need frames_zm_*, the zonal-mean-at-every-level history. Under
# FRAME_LEVELS=probe the frames carry a size axis but no vertical axis, so there
# is nothing to animate here and no approximation worth substituting: say what is
# missing and how to get it rather than drawing the probe level again.
if "frames_zm_mas" not in fr.files:
    print(f"  lat-height gifs SKIPPED: coupled_frames_{TAG}.npz has no zonal "
          f"frames (run made with FRAME_LEVELS=probe).\n"
          f"    Re-run with FRAME_LEVELS=all to get them.", flush=True)
else:
    zh = np.asarray(fr["frames_zm_hours"], float)
    zplev = np.asarray(fr["plev_pa"], float) / 100.0
    zlat = np.asarray(fr["lat"], float)
    nzf = len(zh)
    zmass = fr["frames_zm_mas"].sum(axis=1) * 1e9        # (nzf,nlev,nlat)
    zdT = fr["frames_zm_dT"]                             # (nzf,nlev,nlat)

    # LOG norm for mass, unlike the maps. A zonal mean spans ~4 decades from the
    # tropical plume to the winter polar cap (the maps span well under one), so a
    # linear scale anchored on the plume renders everything else as a flat floor.
    # Clipped to 4 decades so the cap does not eat the whole colormap.
    _hi = float(np.nanpercentile(zmass, 99.8))
    _lo = float(np.nanpercentile(zmass[zmass > 0], 1.0)) if np.any(zmass > 0) else _hi / 1e4
    znorm = mcolors.LogNorm(vmin=max(_lo, _hi / 1e4), vmax=_hi)
    _dpos = max(float(np.nanpercentile(zdT, 99.8)), 1e-6)
    zdnorm = mcolors.Normalize(vmin=-_dpos, vmax=_dpos)

    ZFIELDS = [
        dict(name="zonal_so4", data=zmass, cmap="viridis", norm=znorm,
             label="SO4 [x1e-9 kg/kg]", title="zonal-mean SO4 mass", extend="both"),
        dict(name="zonal_dTrad", data=zdT, cmap="RdBu_r", norm=zdnorm,
             label="dT_rad [K]", title="zonal-mean radiative heating", extend="max"),
    ]

    def render_zonal(field, outdir):
        """One PNG per frame of a latitude-height section. Same single-figure
        reuse as render(): only the mesh array and the title change."""
        dpi = 100.0
        fig = plt.figure(figsize=(args.width / dpi, args.width / dpi * 0.62),
                         dpi=dpi, constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        mesh = ax.pcolormesh(zlat, zplev, field["data"][0], cmap=field["cmap"],
                             norm=field["norm"], shading="auto")
        ax.set_yscale("log"); ax.invert_yaxis()
        ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlim(-90, 90); ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax.set_xlabel("latitude [deg]"); ax.set_ylabel("pressure [hPa]")
        fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.028, pad=0.01,
                     aspect=22, label=field["label"], extend=field["extend"])
        ttl = fig.suptitle("", fontsize=12, fontweight="bold")
        paths = []
        for i in range(nzf):
            mesh.set_array(field["data"][i].ravel())
            ttl.set_text(f"{field['title']}   --   day {zh[i]/24:5.1f}")
            path = os.path.join(outdir, f"f{i:04d}.png")
            fig.savefig(path)
            paths.append(path)
        plt.close(fig)
        return paths

    for field in ZFIELDS:
        out = f"{TAG}_{field['name']}.gif"
        with tempfile.TemporaryDirectory(prefix=f"gif_{field['name']}_") as td:
            render_zonal(field, td)
            assemble(td, out, args.fps, args.hold, n=nzf)
        print(f"wrote {out}  ({nzf} frames, {zh[-1]/24:.0f} days, "
              f"{os.path.getsize(out)/1e6:.1f} MB)", flush=True)
