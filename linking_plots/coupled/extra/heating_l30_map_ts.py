#!/usr/bin/env python3
"""Stratospheric aerosol heating: last-30-day maps + the gradual build-up (2026-11-09, PYC).
Top: last-30-day mean column-integrated aerosol heating (injection minus control), stratospheric levels only, as the
     extra energy absorbed per m2:  cp * sum(dHR * dp/g)  [W/m2]  (dHR in K/s). From agg_v2 l30_*_dhr_kday_colint.
Bottom: global stratospheric mass-weighted mean net heating rate through 1997-1998 (mK/day), from heating_timeseries.npz.
No feedback: the temperature never responds, so 'Delta T' is the heating rate the aerosol adds, not a realised warming."""
import os, sys, datetime
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import cartopy.crs as ccrs

C = '/data/AIDE-SAI-link/linking_plots/coupled'
A = np.load(f'{C}/cache/agg_v2.npz'); T = np.load(f'{C}/extra/heating_timeseries.npz')
lat = A['lat']; lon = np.arange(288) * 1.25; CP = 1004.64
w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 288)); w /= w.sum()
K = ('num', 'emu', 'mam'); LAB = {'emu': 'TOMAS emulator', 'num': 'TOMAS numerical', 'mam': 'MAM4'}
COL = {'emu': '#eb6834', 'num': '#2a78d6', 'mam': '#1baf7a'}
# The saved frames are at 4 fixed UTC times a day, so a cell's SW heating is sampled at 4 fixed LOCAL times, which draws
# day/night arcs. SW aerosol heating is NOT proportional to insolation (negative at low sun, positive at high sun), so an
# insolation rescaling does not remove them (tried 2026-11-09). Instead: LW map as saved (no diurnal sampling issue) +
# the ZONAL MEAN of the SW part, which averages the 4 samples over all longitudes = all local times = the daily mean.
# This drops only the longitudinal structure of the SW part; the global mean is unchanged.
Q = {}
for k in K:
    net = A[f'l30_{k}_dhr_kday_colint']; sw = A[f'l30_{k}_dhr_sw_kday_colint']
    swz = sw.mean(1, keepdims=True) * np.ones((1, 288))
    Q[k] = (net - sw + swz) * CP / 86400                                   # W/m2
    print(f'{k}: LW {((net - sw) * CP / 86400 * w).sum():.4f}  SW {(sw * CP / 86400 * w).sum():+.4f} W/m2 (global means)')
nd = T['num_net'].shape[0]; D0 = datetime.date(1997, 1, 1)
lo = float(np.percentile(np.concatenate([Q[k].ravel() for k in K]), 0.5)); hi = float(np.percentile(np.concatenate([Q[k].ravel() for k in K]), 99.5))
lim = max(abs(lo), hi)
plt.rcParams.update({'font.size': 9.5})
fig = plt.figure(figsize=(16, 8.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.28, wspace=0.06, left=0.04, right=0.98, top=0.9, bottom=0.1)
cmap = plt.get_cmap('RdBu_r')
for j, k in enumerate(K):
    ax = fig.add_subplot(gs[0, j], projection=ccrs.PlateCarree(central_longitude=180)); ax.coastlines(lw=0.4, color='0.3'); ax.set_global()
    im = ax.pcolormesh(lon, lat, Q[k], cmap=cmap, norm=mcolors.TwoSlopeNorm(0, -lim, lim), shading='auto', transform=ccrs.PlateCarree(), rasterized=True)
    ax.plot([180, 180], [29.7, -29.7], 'k+', ms=7, mew=1.2, transform=ccrs.PlateCarree(), ls='none')
    ax.set_title(f'{LAB[k]}: global mean {(Q[k] * w).sum():.3f} W m$^{{-2}}$', fontsize=9.5)
cb = fig.colorbar(im, ax=fig.axes[:3], orientation='horizontal', shrink=0.45, pad=0.05, aspect=40, extend='both')
cb.set_label('extra energy absorbed in the stratospheric column by the aerosol, last 30 days (Dec 1998), W m$^{-2}$ (LW map + zonal-mean SW, see note)')
a = fig.add_subplot(gs[1, :])
xd = [D0 + datetime.timedelta(days=i) for i in range(nd)]
run = lambda x: np.convolve(x, np.ones(30) / 30, mode='same')
for k in K:
    y = 1e3 * T[f'{k}_net']
    a.plot(xd, y, color=COL[k], lw=0.6, alpha=0.35); a.plot(xd[15:-15], run(y)[15:-15], color=COL[k], lw=2, label=LAB[k])
    print(f'{k}: last-30 mean net {y[-30:].mean():.2f} mK/day; column map global mean {(Q[k] * w).sum():.4f} W/m2; '
          f'map p99.5 {np.percentile(Q[k], 99.5):.3f}, min {Q[k].min():.3f}')
a.axvspan(xd[-30], xd[-1], color='0.85', zorder=0, label='last 30 days (maps)')
a.axhline(0, color='0.6', lw=0.6); a.grid(alpha=0.25); a.legend(frameon=False, loc='upper left')
a.set_ylabel('net aerosol heating rate, mK day$^{-1}$'); a.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
a.set_xlim(xd[0], xd[-1])
a.set_title('The gradual heating: global stratospheric mass-weighted mean net (SW + LW) aerosol heating rate, injection minus control '
            '(thin daily, thick 30-day running mean)', loc='left', fontsize=9.5)
fig.suptitle('Stratospheric aerosol heating (ΔT as a heating rate; no feedback, so the temperature itself does not respond)', fontsize=11)
fig.text(0.5, 0.012, 'Maps: longwave part per cell; shortwave part as its zonal mean (the 6-hourly fixed-UTC frames sample each cell at only 4 local times, which otherwise draws day/night arcs)', ha='center', fontsize=8.5, color='0.35')
out = f'{C}/extra/heating_l30_map_timeseries.png'; fig.savefig(out, dpi=150); print('wrote', out)
