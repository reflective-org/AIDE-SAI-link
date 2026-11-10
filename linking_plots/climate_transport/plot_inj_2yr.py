#!/usr/bin/env python3
"""Two-panel GIF of the 1997-1998 passive injection tracer: column mass map (left), zonal-mean
mixing ratio latitude x pressure (right). Input data/inj2yr.npz from inj_transport_2yr.py.
Fixed log colour limits for all frames, set from the last day (99.99th pct, 4 decades), as in year 1."""
import io, os, datetime
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(f'{HERE}/data/inj2yr.npz')
col, zm, lat, lon, plev, bud, cells = Z['inj_col'], Z['inj_zm'] * 1e9, Z['lat'], Z['lon'], Z['plev'] / 100, Z['budget'], Z['inj_cells']
nd = col.shape[0]; D0 = datetime.date(1997, 1, 1)
col_hi = float(10 ** np.ceil(np.log10(np.percentile(col[-1], 99.99)))); col_lo = col_hi * 1e-4
zm_hi = float(10 ** np.ceil(np.log10(np.percentile(zm[-1], 99.99)))); zm_lo = zm_hi * 1e-4
print(f'limits: column {col_lo:.1e}-{col_hi:.1e} kg/m2, zonal mean {zm_lo:.1e}-{zm_hi:.1e} ppbm')

fig = plt.figure(figsize=(15, 5.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.08, left=0.01, right=0.97, top=0.84, bottom=0.12)
ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree(central_longitude=180)); ax.coastlines(lw=0.5, color='0.3'); ax.set_global()
im1 = ax.pcolormesh(lon, lat, np.clip(col[0], col_lo, None), cmap='Blues', norm=LogNorm(col_lo, col_hi), shading='nearest',
                    transform=ccrs.PlateCarree(), rasterized=True)
for c in cells: ax.plot(c[2], c[1], 'x', color='#c2452f', ms=7, mew=2, transform=ccrs.PlateCarree())
fig.colorbar(im1, ax=ax, orientation='horizontal', shrink=0.7, pad=0.04, extend='min').set_label('column tracer mass, kg m$^{-2}$ (log)')
ax.set_title('Column mass (Σ q·dp/g, 1.2-143 hPa)', fontsize=10)
a2 = fig.add_subplot(gs[1])
im2 = a2.pcolormesh(lat, plev, np.clip(zm[0], zm_lo, None), cmap='Blues', norm=LogNorm(zm_lo, zm_hi), shading='nearest', rasterized=True)
a2.set_yscale('log'); a2.set_ylim(plev.max(), plev.min()); a2.set_yticks([100, 50, 20, 10, 5, 2]); a2.set_yticklabels(['100', '50', '20', '10', '5', '2'])
a2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a2.set_xlim(-90, 90); a2.set_xticks(range(-90, 91, 30))
a2.set_xlabel('latitude'); a2.set_ylabel('hPa')
for c in cells: a2.plot(c[1], c[0], 'x', color='#c2452f', ms=7, mew=2)
fig.colorbar(im2, ax=a2, orientation='horizontal', shrink=0.8, pad=0.12, extend='min').set_label('zonal-mean mixing ratio, ppbm (log)')
a2.set_title('Zonal mean, latitude × pressure', fontsize=10)
sup = fig.suptitle('', fontsize=11.5)

frames = []
days = list(range(0, nd, 4)) + ([nd - 1] if (nd - 1) % 4 else [])
for d in days:
    im1.set_array(np.clip(col[d], col_lo, None).ravel()); im2.set_array(np.clip(zm[d], zm_lo, None).ravel())
    _, rel, inb = bud[d]
    sup.set_text(f'Passive injection tracer on emulator winds (10 Tg/yr at 52 hPa, 30°N/S, 180°E), '
                 f'{(D0 + datetime.timedelta(days=d)).strftime("%d %b %Y")} (day {d + 1})\n'
                 f'released {rel / 1e9:.2f} Tg, still in the 1.2-143 hPa domain {inb / 1e9:.2f} Tg ({100 * inb / rel:.1f} %)')
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=80); b.seek(0); frames.append(Image.open(b).convert('RGB'))
    if d == nd - 1: fig.savefig(f'{HERE}/figs/inj2yr_final_day.png', dpi=120)
idx = np.linspace(0, len(frames) - 1, 8).astype(int); W, H = frames[0].size
canvas = Image.new('RGB', (W, H * len(idx)))
for k, i in enumerate(idx): canvas.paste(frames[i], (0, k * H))
pal = canvas.quantize(colors=200, method=Image.Quantize.MAXCOVERAGE)
fr = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
out = f'{HERE}/figs/inj2yr_map_zonal.gif'
fr[0].save(out, save_all=True, append_images=fr[1:], duration=120, loop=0)
print('wrote', out, len(fr), 'frames', f'{os.path.getsize(out) / 1e6:.1f} MB')
for dd in (364, nd - 1):
    print(f'day {dd + 1}: released {bud[dd][1] / 1e9:.3f} Tg, in domain {bud[dd][2] / 1e9:.3f} Tg ({100 * bud[dd][2] / bud[dd][1]:.1f} %)')
