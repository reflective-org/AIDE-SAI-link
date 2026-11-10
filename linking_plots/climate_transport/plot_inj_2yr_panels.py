#!/usr/bin/env python3
"""Static panels of the 1997-1998 passive injection tracer at days 30, 90, 180, 365, 545, 730 (PYC).
Each row: column mass map (left) | zonal-mean mixing ratio, latitude x pressure (right). Same fixed log limits as the GIF
(plot_inj_2yr.py: last day 99.99th percentile, 4 decades). Input data/inj2yr.npz."""
import os, datetime
import numpy as np
import matplotlib.ticker
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs

HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(f'{HERE}/data/inj2yr.npz')
col, zm, lat, lon, plev, bud, cells = Z['inj_col'], Z['inj_zm'] * 1e9, Z['lat'], Z['lon'], Z['plev'] / 100, Z['budget'], Z['inj_cells']
col_hi = float(10 ** np.ceil(np.log10(np.percentile(col[-1], 99.99)))); col_lo = col_hi * 1e-4
zm_hi = float(10 ** np.ceil(np.log10(np.percentile(zm[-1], 99.99)))); zm_lo = zm_hi * 1e-4
DAYS = (30, 90, 180, 365, 545, 730); D0 = datetime.date(1997, 1, 1)
fig = plt.figure(figsize=(14, 21))
gs = fig.add_gridspec(len(DAYS), 2, width_ratios=[1.55, 1], hspace=0.32, wspace=0.08, left=0.03, right=0.9, top=0.935, bottom=0.05)
for r, day in enumerate(DAYS):
    d = day - 1; _, rel, inb = bud[d]
    ax = fig.add_subplot(gs[r, 0], projection=ccrs.PlateCarree(central_longitude=180)); ax.coastlines(lw=0.5, color='0.3'); ax.set_global()
    im1 = ax.pcolormesh(lon, lat, np.clip(col[d], col_lo, None), cmap='Blues', norm=LogNorm(col_lo, col_hi), shading='nearest',
                        transform=ccrs.PlateCarree(), rasterized=True)
    for c in cells: ax.plot(c[2], c[1], 'x', color='#c2452f', ms=6, mew=1.8, transform=ccrs.PlateCarree())
    ax.set_title(f'Day {day} ({(D0 + datetime.timedelta(days=d)):%d %b %Y}): released {rel / 1e9:.2f} Tg, in domain {inb / 1e9:.2f} Tg '
                 f'({100 * inb / rel:.1f} %)', fontsize=10, loc='left')
    a2 = fig.add_subplot(gs[r, 1])
    im2 = a2.pcolormesh(lat, plev, np.clip(zm[d], zm_lo, None), cmap='Blues', norm=LogNorm(zm_lo, zm_hi), shading='nearest', rasterized=True)
    a2.set_yscale('log'); a2.set_ylim(plev.max(), plev.min()); a2.set_yticks([100, 50, 20, 10, 5, 2]); a2.set_yticklabels(['100', '50', '20', '10', '5', '2'])
    a2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a2.set_xlim(-90, 90); a2.set_xticks(range(-90, 91, 30)); a2.set_ylabel('hPa')
    for c in cells: a2.plot(c[1], c[0], 'x', color='#c2452f', ms=6, mew=1.8)
    if r == len(DAYS) - 1: a2.set_xlabel('latitude')
    print(f'day {day}: released {rel / 1e9:.3f} Tg, in domain {inb / 1e9:.3f} Tg ({100 * inb / rel:.1f} %)')
c1 = fig.add_axes([0.06, 0.03, 0.45, 0.006]); fig.colorbar(im1, cax=c1, orientation='horizontal', extend='min').set_label('column tracer mass, kg m$^{-2}$ (log)')
c2 = fig.add_axes([0.92, 0.25, 0.012, 0.5]); fig.colorbar(im2, cax=c2, extend='min').set_label('zonal-mean mixing ratio, ppbm (log)')
fig.suptitle('Passive injection tracer on emulator winds, 1997-1998 (10 Tg/yr at 52 hPa, 30°N/S, 180°E; transport only, campaign advection)\n'
             'left: column mass, 1.2-143 hPa; right: zonal-mean mixing ratio; fixed colour limits', fontsize=11.5)
out = f'{HERE}/figs/inj2yr_panels_d30_90_180_365_545_730.png'; fig.savefig(out, dpi=120); print('wrote', out)
