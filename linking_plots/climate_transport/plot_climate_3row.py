#!/usr/bin/env python3
"""Three rows: CESM | emulator | emulator - CESM, columns T, u, v, omega (annual-mean zonal means 1997)."""
import numpy as np, matplotlib.pyplot as plt
from plotlib import DIV, latp
OUT = '/data/AIDE-SAI-link/linking_plots/climate_transport'
d = np.load(f'{OUT}/data/climate_zm.npz'); lat = d['lat']; p = d['plev'] / 100
am = lambda x: x[:-1].mean(0)
cols = [('T', 'T [K]', 1), ('u', 'u [m/s]', 1), ('v', 'v [m/s]', 1), ('omega', 'omega [mPa/s]', 1e3)]
fig, ax = plt.subplots(3, 4, figsize=(20, 11.5))
for j, (k, lab, sc) in enumerate(cols):
    c, e = am(d['cesm_' + k]) * sc, am(d['emu_' + k]) * sc
    if k == 'T':
        lo = np.floor(min(np.percentile(c, .5), np.percentile(e, .5))); hi = np.ceil(max(np.percentile(c, 99.5), np.percentile(e, 99.5)))
        kw = dict(cmap='viridis', vmin=lo, vmax=hi, contours=np.arange(180, 300, 10))
    else:
        L = max(np.percentile(np.abs(c), 99), np.percentile(np.abs(e), 99))
        L = float(np.ceil(L / 10) * 10) if k == 'u' else float(L)
        kw = dict(cmap=DIV, vmin=-L, vmax=L, contours=np.arange(-100, 101, 10) if k == 'u' else None)
    for i, (f, name) in enumerate([(c, 'CESM'), (e, 'emulator')]):
        latp(ax[i, j], lat, p, f, kw['cmap'], kw['vmin'], kw['vmax'], lab, fig, contours=kw['contours'], ylabel=(j == 0))
        ax[i, j].set_title(f'{name}: {lab}, annual mean 1997')
    da = e - c; lim = float(np.percentile(np.abs(da), 99)); lim = float(np.ceil(lim)) if lim > 2 else lim
    latp(ax[2, j], lat, p, da, DIV, -lim, lim, 'emulator - CESM ' + lab, fig, ylabel=(j == 0))
    ax[2, j].set_title(f'emulator - CESM: {lab}')
fig.suptitle('Annual-mean zonal means, 1997: CESM (top), JCM emulator P2@200 (middle), emulator minus CESM (bottom). '
             'CESM and emulator share colour limits; omega > 0 = descent.', fontsize=11)
fig.tight_layout(); fig.savefig(f'{OUT}/climate_annual_mean_cesm_emu_diff_3row.png', dpi=130)
