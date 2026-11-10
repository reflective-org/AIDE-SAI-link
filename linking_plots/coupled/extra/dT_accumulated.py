#!/usr/bin/env python3
"""How much the radiation 'should have' changed T in each run, with nothing taking it back (2026-11-09, PY).
The runs are RAD_MODE=flux: T is never updated. Here each run's own absolute radiative heating rate (stratospheric cells,
the injection run's tropopause mask applied to both runs, as in process.py) is integrated in time with NO damping and no
dynamics:  dT_run(t) = sum_{days <= t} HR_run(day) * 1 day.  Done separately for the injection and its control run, then
differenced. The control's own accumulation is the net radiative imbalance the prescribed circulation balances in reality;
the injection-minus-control accumulation is the aerosol's undamped heating.
Inputs: cache/day_DDD.npz, day_mam_DDD.npz ({k}_hr_inj_zsum, {k}_hr_ctl_zsum, strat_zcount). Output: extra/dT_accumulated.png/.npz"""
import os, sys, datetime
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.colors as mcolors, matplotlib.dates as mdates
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from common import CACHE, OUT, NDAYS, frame_file

S = [('num', '#2a78d6', 'TOMAS numerical'), ('emu', '#eb6834', 'TOMAS emulator'), ('mam', '#1baf7a', 'MAM4')]
lat = np.load(f'{CACHE}/timeseries.npz')['lat']
plev = np.load(frame_file('r2_tomas_fast', 400))['plev_pa'] / 100
dp = np.abs(np.gradient(plev))[:, None]; w = dp * np.cos(np.deg2rad(lat))[None, :]
G = {f'{m}_{r}': np.zeros(NDAYS) for m, _, _ in S for r in ('inj', 'ctl')}      # global strat mass-weighted mean, K/day
Z = {f'{m}_{r}': np.zeros((NDAYS, plev.size, lat.size)) for m, _, _ in S for r in ('inj', 'ctl')}  # zonal mean, K/day
for d in range(NDAYS):
    z = np.load(f'{CACHE}/day_{d:03d}.npz'); zm = np.load(f'{CACHE}/day_mam_{d:03d}.npz')
    for m, src in (('emu', z), ('num', z), ('mam', zm)):
        cnt = src['strat_zcount']
        for r in ('inj', 'ctl'):
            s = src[f'{m}_hr_{r}_zsum']
            G[f'{m}_{r}'][d] = (s * w).sum() / (cnt * w).sum()
            Z[f'{m}_{r}'][d] = np.where(cnt > 0.5, s / np.maximum(cnt, 1e-9), np.nan)
C = {k: np.cumsum(v) for k, v in G.items()}                                    # K, undamped accumulation
ZC = {m: np.nansum(Z[f'{m}_inj'] - Z[f'{m}_ctl'], 0) for m, _, _ in S}        # 2-year accumulated inj - ctl, lat x p
ZC = {m: np.where(np.isfinite(Z[f'{m}_inj'][-1]), v, np.nan) for m, v in ZC.items()}

y1, y2 = slice(0, 365), slice(365, 730)
print('global stratospheric mass-weighted mean; undamped accumulation of each run\'s own heating rate')
for m, _, nm in S:
    hi, hc = G[f'{m}_inj'], G[f'{m}_ctl']; dd = hi - hc
    print(f'{nm:16s}: mean HR inj {hi.mean():+.4f} ctl {hc.mean():+.4f} K/day | accumulated after 1997: inj {hi[y1].sum():+.2f} ctl {hc[y1].sum():+.2f} '
          f'diff {dd[y1].sum():+.3f} K | after 1998 alone: diff {dd[y2].sum():+.3f} K | 2-year total: inj {hi.sum():+.2f} ctl {hc.sum():+.2f} '
          f'diff {dd.sum():+.3f} K | last-30-day mean diff rate {1e3 * dd[-30:].mean():.2f} mK/day | lat-p max {np.nanmax(ZC[m]):.1f} K')

xd = [datetime.date(1997, 1, 1) + datetime.timedelta(days=i) for i in range(NDAYS)]
plt.rcParams.update({'font.size': 9.5})
fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.15], hspace=0.42, wspace=0.12, left=0.06, right=0.93, top=0.93, bottom=0.05)
a = fig.add_subplot(gs[0, :])
for m, c, nm in S:
    a.plot(xd, G[f'{m}_inj'], color=c, lw=1.4, label=f'{nm}, injection'); a.plot(xd, G[f'{m}_ctl'], color=c, lw=1.0, ls='--', label=f'{nm}, control')
a.set_ylabel('K day$^{-1}$'); a.grid(alpha=0.25); a.legend(frameon=False, ncol=3, fontsize=8, loc='lower left'); a.axhline(0, color='0.6', lw=0.6)
a.set_title('(a) Absolute radiative heating rate in each run (global stratospheric mass-weighted mean, daily)', loc='left')
b = fig.add_subplot(gs[1, :], sharex=a)
for m, c, nm in S:
    b.plot(xd, C[f'{m}_inj'], color=c, lw=1.4, label=f'{nm}, injection'); b.plot(xd, C[f'{m}_ctl'], color=c, lw=1.0, ls='--', label=f'{nm}, control')
b.set_ylabel('K'); b.grid(alpha=0.25); b.axhline(0, color='0.6', lw=0.6)
b.set_title('(b) How far each run\'s temperature would have moved: the heating rate summed day by day, no damping, no dynamics', loc='left')
c_ = fig.add_subplot(gs[2, :], sharex=a)
for m, c, nm in S:
    c_.plot(xd, C[f'{m}_inj'] - C[f'{m}_ctl'], color=c, lw=2, label=nm)
c_.axvline(xd[364], color='0.5', lw=0.8, ls=':'); c_.set_ylabel('K'); c_.grid(alpha=0.25); c_.legend(frameon=False, loc='upper left')
c_.set_title('(c) Injection minus control: the accumulated aerosol ΔT (undamped)', loc='left')
c_.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y')); c_.set_xlim(xd[0], xd[-1])
plt.setp(a.get_xticklabels(), visible=False); plt.setp(b.get_xticklabels(), visible=False)
L = float(np.nanpercentile(np.abs(np.concatenate([v.ravel() for v in ZC.values()])), 99.5))
cm = plt.get_cmap('RdBu_r').copy(); cm.set_bad('0.85')
for j, (m, _, nm) in enumerate(S):
    ax = fig.add_subplot(gs[3, j])
    im = ax.pcolormesh(lat, plev, ZC[m], cmap=cm, norm=mcolors.Normalize(-L, L), shading='nearest', rasterized=True)
    ax.set_yscale('log'); ax.set_ylim(plev.max(), plev.min()); ax.set_yticks([100, 50, 20, 10, 5, 2]); ax.set_yticklabels(['100', '50', '20', '10', '5', '2'])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks(range(-90, 91, 30)); ax.set_xlabel('latitude')
    if j == 0: ax.set_ylabel('hPa')
    ax.set_title(f'(d) {nm}: 2-year accumulated ΔT, zonal mean', fontsize=9.5)
fig.colorbar(im, ax=fig.axes[-3:], fraction=0.015, pad=0.01, extend='both').set_label('K (undamped)')
fig.suptitle('Radiative temperature change implied by the heating rates, with nothing taking it back (the runs never update T)', fontsize=11)
fig.savefig(f'{OUT}/extra/dT_accumulated.png', dpi=140); print('wrote', f'{OUT}/extra/dT_accumulated.png')
np.savez(f'{OUT}/extra/dT_accumulated.npz', plev=plev, lat=lat, **{f'G_{k}': v for k, v in G.items()}, **{f'C_{k}': v for k, v in C.items()},
         **{f'ZC_{m}': v for m, v in ZC.items()})
