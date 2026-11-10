import datetime as dt
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

DIV = LinearSegmentedColormap.from_list(
    'bgr', ['#08306b', '#2166ac', '#6baed6', '#d4d4d4', '#fc9272', '#cb181d', '#67000d'])
T0 = dt.datetime(1997, 1, 1, 6)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10})


def date_of_day(d):
    return (T0 + dt.timedelta(days=float(d))).strftime('%Y-%m-%d')


def latp_axes(ax, plev_hpa, ylabel=True):
    ax.set_yscale('log')
    ax.set_ylim(plev_hpa.max() * 1.03, plev_hpa.min() / 1.03)
    ticks = [p for p in (1, 2, 5, 10, 20, 50, 100, 140) if plev_hpa.min() / 1.05 <= p <= plev_hpa.max() * 1.05]
    ax.set_yticks(ticks); ax.set_yticklabels([str(t) for t in ticks]); ax.minorticks_off()
    ax.set_xlim(-90, 90); ax.set_xticks(range(-90, 91, 30))
    ax.set_xticklabels(['90S', '60S', '30S', 'EQ', '30N', '60N', '90N'])
    if ylabel:
        ax.set_ylabel('pressure [hPa]')


def latp(ax, lat, plev_hpa, f, cmap, vmin, vmax, cbar_label, fig, contours=None, ylabel=True, norm=None):
    kw = dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)
    m = ax.pcolormesh(lat, plev_hpa, f, cmap=cmap, shading='nearest', rasterized=True, **kw)
    if contours is not None:
        cs = ax.contour(lat, plev_hpa, f, levels=contours, colors='k', linewidths=0.5, alpha=0.6)
        ax.clabel(cs, fontsize=7, fmt='%g')
    latp_axes(ax, plev_hpa, ylabel)
    cb = fig.colorbar(m, ax=ax, pad=0.02)
    cb.set_label(cbar_label)
    return m


def fig_to_img(fig, dpi):
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=dpi); buf.seek(0)
    return Image.open(buf).convert('RGB')


def save_gif(frames, path, ms=120, colors=200):
    """One palette shared by every frame (built from frames spread over the sequence), so a colour
    means the same value in every frame; per-frame palettes made fixed colour bars look like they drift."""
    import numpy as _np
    w, h = frames[0].size
    pick = sorted(set(_np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)))
    strip = Image.new('RGB', (w, h * len(pick)))
    for k, j in enumerate(pick):
        strip.paste(frames[j].resize((w, h)), (0, h * k))
    pal = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=ms, loop=0)


def coastlines(path='/data/AIDE-SAI-link/linking_plots/assets/ne_110m/ne_110m_coastline.shp'):
    """Minimal polyline .shp reader -> list of (lon, lat) arrays, lon in [0, 360)."""
    import struct
    b = open(path, 'rb').read(); pos = 100; out = []
    while pos < len(b):
        _, clen = struct.unpack('>ii', b[pos:pos + 8]); pos += 8
        rec = b[pos:pos + 2 * clen]; pos += 2 * clen
        st = struct.unpack('<i', rec[:4])[0]
        if st not in (3, 13, 23):
            continue
        npart, npt = struct.unpack('<ii', rec[36:44])
        parts = list(struct.unpack(f'<{npart}i', rec[44:44 + 4 * npart])) + [npt]
        pts = np.frombuffer(rec[44 + 4 * npart:44 + 4 * npart + 16 * npt], '<f8').reshape(-1, 2)
        for a_, b_ in zip(parts[:-1], parts[1:]):
            xy = pts[a_:b_].copy(); xy[:, 0] %= 360
            # split where the line wraps across 0/360
            br = np.where(np.abs(np.diff(xy[:, 0])) > 180)[0] + 1
            out.extend(np.split(xy, br))
    return out
