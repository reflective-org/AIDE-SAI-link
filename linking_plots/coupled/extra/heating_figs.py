#!/usr/bin/env python3
"""Aerosol heating (dT shown as heating rate, injection - control) for the three schemes, from the per-day caches
(cache/day_DDD.npz = emulator + numerical, cache/day_mam_DDD.npz = MAM4; written by scripts/process.py).
Rebuilt 2026-11-09 from the inline code that made the published figures:
  fig3_d_heating_timeseries.png : global mass-weighted strat mean, SW / LW / net + difference vs numerical (mK/day)
  fig3_e_heating_lat_season.png : latitude x day, 3 schemes + emulator-numerical, SW / LW / net (5-day smoothing)
  fig3_f_heating_bands.png      : six latitude bands, numerical (solid) vs MAM4 (dotted), 15-day centred mean
Writes heating_timeseries.npz and heating_lat_day.npz next to the figures. usage: heating_figs.py"""
import os, sys
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from common import CACHE, OUT, NDAYS, frame_file

EX = f'{OUT}/extra'
DIV = LinearSegmentedColormap.from_list('bgr', ['#08306b', '#2166ac', '#6baed6', '#d4d4d4', '#fc9272', '#cb181d', '#67000d'])
S = [('emu', '#eb6834', 'TOMAS emulator'), ('num', '#2a78d6', 'TOMAS numerical'), ('mam', '#1baf7a', 'MAM4')]
YLAB = os.environ.get('LP_DAYLABEL', 'day since 1997-01-01' if NDAYS > 365 else 'day of 1997')

lat = np.load(f'{CACHE}/timeseries.npz')['lat']
plev = np.load(frame_file('r2_tomas_fast', 400))['plev_pa']
dp = np.abs(np.gradient(plev))[:, None]
w = dp * np.cos(np.deg2rad(lat))[None, :]
o = {f'{m}_{k}': np.zeros(NDAYS) for m, _, _ in S for k in ('net', 'sw')}
H = {f'{m}_{k}': np.zeros((NDAYS, lat.size)) for m, _, _ in S for k in ('net', 'sw')}
for d in range(NDAYS):
    z = np.load(f'{CACHE}/day_{d:03d}.npz'); zm = np.load(f'{CACHE}/day_mam_{d:03d}.npz')
    for m, src in (('emu', z), ('num', z), ('mam', zm)):
        cnt = src['strat_zcount']
        o[f'{m}_net'][d] = (src[f'{m}_dhr_kday_zsum'] * w).sum() / (cnt * w).sum()
        o[f'{m}_sw'][d] = (src[f'{m}_dhr_sw_kday_zsum'] * w).sum() / (cnt * w).sum()
        den = (cnt * dp).sum(0)
        H[f'{m}_net'][d] = (src[f'{m}_dhr_kday_zsum'] * dp).sum(0) / den
        H[f'{m}_sw'][d] = (src[f'{m}_dhr_sw_kday_zsum'] * dp).sum(0) / den
for m, _, _ in S:
    o[f'{m}_lw'] = o[f'{m}_net'] - o[f'{m}_sw']; H[f'{m}_lw'] = H[f'{m}_net'] - H[f'{m}_sw']
np.savez(f'{EX}/heating_timeseries.npz', **o); np.savez(f'{EX}/heating_lat_day.npz', lat=lat, **H)
day = np.arange(1, NDAYS + 1)

# fig3_d
run = lambda x: np.convolve(x, np.ones(30) / 30, 'valid'); dc = day[14:14 + NDAYS - 29]
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(4, 1, figsize=(10, 11.5), sharex=True)
for a, (k, t) in zip(ax[:3], (('sw', 'shortwave'), ('lw', 'longwave'), ('net', 'net (SW + LW)'))):
    for m, c, lab in S:
        a.plot(day, o[f'{m}_{k}'] * 1e3, color=c, lw=0.6, alpha=0.4)
        a.plot(dc, run(o[f'{m}_{k}']) * 1e3, color=c, lw=2, label=lab)
    a.axhline(0, color='0.6', lw=0.6); a.set_ylabel('mK / day'); a.set_title(f'Aerosol heating rate, {t}', loc='left')
ax[0].legend(frameon=False, loc='lower left')
for m, c, lab in S[::2]:
    for k, ls in (('net', '-'), ('lw', '--'), ('sw', ':')):
        ax[3].plot(dc, 1e3 * (run(o[f'{m}_{k}']) - run(o[f'num_{k}'])), color=c, ls=ls, lw=1.5,
                   label=f'{lab.replace("TOMAS ", "")} {k.upper() if k != "net" else "net"}')
ax[3].axhline(0, color='0.6', lw=0.6); ax[3].set_ylabel('mK / day'); ax[3].legend(frameon=False, ncol=2, fontsize=9)
ax[3].set_title('Minus TOMAS numerical (30-day running means)', loc='left'); ax[3].set_xlabel(YLAB)
fig.suptitle('ΔT as heating rate: global mass-weighted mean over stratospheric cells, injection minus control\n'
             'thin = daily, thick = centred 30-day running mean', fontsize=11)
fig.tight_layout(); fig.savefig(f'{EX}/fig3_d_heating_timeseries.png', dpi=150); plt.close(fig)
for m, _, _ in S:
    print(m, 'last30 net %.2f sw %.2f lw %.2f mK/day' % tuple(1e3 * o[f'{m}_{k}'][-30:].mean() for k in ('net', 'sw', 'lw')))

# fig3_e
sm = lambda x: np.apply_along_axis(lambda c: np.convolve(c, np.ones(5) / 5, 'same'), 0, x)
names = [(m, lab) for m, _, lab in S]
plt.rcParams.update({'font.size': 9})
fig, ax = plt.subplots(3, 4, figsize=(18, 10.5), sharex=True, sharey=True)
for r, (k, t) in enumerate((('sw', 'shortwave'), ('lw', 'longwave'), ('net', 'net'))):
    L = max(np.nanpercentile(np.abs(H[f'{m}_{k}']), 99.5) for m, _ in names) * 1e3
    for c, (m, nm) in enumerate(names):
        im = ax[r, c].pcolormesh(day, lat, sm(H[f'{m}_{k}']).T * 1e3, cmap=DIV, vmin=-L, vmax=L, shading='nearest', rasterized=True)
        ax[r, c].set_title(f'{nm}: {t} heating', fontsize=10)
    fig.colorbar(im, ax=ax[r, :3], pad=0.01, extend='both').set_label('mK / day')
    dif = sm(H[f'emu_{k}'] - H[f'num_{k}']).T * 1e3; Ld = np.nanpercentile(np.abs(dif), 99.5)
    im = ax[r, 3].pcolormesh(day, lat, dif, cmap=DIV, vmin=-Ld, vmax=Ld, shading='nearest', rasterized=True)
    ax[r, 3].set_title(f'emulator minus numerical: {t}', fontsize=10)
    fig.colorbar(im, ax=ax[r, 3], pad=0.02, extend='both').set_label('mK / day')
for a in ax[:, 0]: a.set_ylabel('latitude'); a.set_yticks(range(-90, 91, 30))
for a in ax[-1]: a.set_xlabel(YLAB)
fig.suptitle('Aerosol heating rate by latitude and season (injection minus control): mass-weighted mean over each latitude\'s '
             'stratospheric levels, zonal mean, 5-day smoothing', fontsize=11)
fig.savefig(f'{EX}/fig3_e_heating_lat_season.png', dpi=130, bbox_inches='tight'); plt.close(fig)

# fig3_f
cw = np.cos(np.deg2rad(lat)); N = 15
runb = lambda x: np.convolve(x, np.ones(N) / N, 'valid'); dcb = day[N // 2: N // 2 + NDAYS - N + 1]
bands = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]
cols = ['#08306b', '#4292c6', '#9ecae1', '#fdae6b', '#e6550d', '#7f2704']
plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
for a, (k, t) in zip(ax, (('sw', 'Shortwave'), ('lw', 'Longwave'), ('net', 'Net'))):
    for (lo, hi), c in zip(bands, cols):
        s = (lat >= lo) & (lat < hi)
        for m, ls in (('num', '-'), ('mam', ':')):
            y = (H[f'{m}_{k}'][:, s] * cw[s]).sum(1) / cw[s].sum() * 1e3
            a.plot(dcb, runb(y), color=c, ls=ls, lw=1.6, label=f'{lo}° to {hi}°' if m == 'num' else None)
    a.axhline(0, color='0.6', lw=0.6); a.set_ylabel('mK / day'); a.set_title(f'{t} aerosol heating by latitude band', loc='left')
    a.spines[['top', 'right']].set_visible(False)
ax[0].legend(frameon=False, ncol=3, fontsize=9, title='solid = TOMAS numerical, dotted = MAM4 (15-day running mean)')
ax[-1].set_xlabel(YLAB); fig.tight_layout(); fig.savefig(f'{EX}/fig3_f_heating_bands.png', dpi=140); plt.close(fig)
print('wrote fig3_d/e/f in', EX)
