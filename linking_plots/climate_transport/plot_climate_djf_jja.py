#!/usr/bin/env python3
"""DJF and JJA zonal-mean T and u, averaged over 1997-1998, 1-300 hPa (2026-11-09).
Sources (data/czm2_*.npz from climate_zm_2yr.py / era5_zm_2yr.py):
  emu  P2@200 emulator | hlc 'hlcorpus' JCM dycore (P2's training corpus) | c1 'jcm_p20bt_c1' JCM dycore (base model's
  training corpus) | cesm CESM2 FWHIST fixedSST | era5 ERA5.
Figures (figs/): climate2yr_{T,u}_{DJF,JJA}.png: top row the 5 sources (shared colour scale, each on its own latitude
grid), bottom row emulator minus each reference (reference interpolated to the emulator latitudes; ERA5 linearly in
log p from its 18 levels; shared symmetric scale), and data/climate2yr_stats.txt (cos-lat weighted RMS / mean differences)."""
import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
D = '/data/AIDE-SAI-link/linking_plots/climate_transport/data'; F = '/data/AIDE-SAI-link/linking_plots/climate_transport/figs'
os.makedirs(F, exist_ok=True)
SRC = [('emu', 'P2@200 emulator'), ('hlc', 'JCM dycore: hlcorpus\n(P2 training data; ERA5 p≥417 hPa, HL drag)'),
       ('c1', 'JCM dycore: jcm_p20bt_c1\n(base training data; ERA5 + QBO nudged)'), ('cesm', 'CESM2 (FWHIST fixedSST)'), ('era5', 'ERA5')]
Z = {k: dict(np.load(f'{D}/czm2_{k}.npz')) for k, _ in SRC}
P = Z['emu']['p_hpa']; LE = Z['emu']['lat']
DIV = mcolors.LinearSegmentedColormap.from_list('bgr', ['#0b3c78', '#2f6fb3', '#8fb6dd', '#e3e3e1', '#eba08a', '#c2452f', '#6d1209'])
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 8.5})
SEAS = {'DJF': (12, 1, 2), 'JJA': (6, 7, 8)}


def season_mean(k, var, s):
    z = Z[k]; mon = np.array([int(d[5:7]) for d in z['dates']])
    x = np.nanmean(z[var][np.isin(mon, SEAS[s])], 0)            # (nlev, nlat)
    p = z['p_hpa']
    if k == 'era5':                                              # 18 native levels -> the 40-level grid, linear in log p
        x = np.stack([np.interp(np.log(P), np.log(p), x[:, j]) for j in range(x.shape[1])], 1); p = P
    assert np.allclose(p, P), k
    return x, z['lat']


def on_emu_lat(x, lat):
    return np.stack([np.interp(LE, lat, x[i]) for i in range(x.shape[0])])


def wrms(d):
    w = np.cos(np.deg2rad(LE))[None] * np.ones_like(d); m = np.isfinite(d)
    return float(np.sqrt((w[m] * d[m] ** 2).sum() / w[m].sum())), float((w[m] * d[m]).sum() / w[m].sum())


def axp(ax):
    ax.set_yscale('log'); ax.set_ylim(300, 1); ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200, 300])
    ax.set_yticklabels(['1', '2', '5', '10', '20', '50', '100', '200', '300']); ax.set_xlim(-90, 90); ax.set_xticks([-60, -30, 0, 30, 60])


stats = []
for var, unit in (('T', 'K'), ('u', 'm s$^{-1}$')):
    for s in SEAS:
        M = {k: season_mean(k, var, s) for k, _ in SRC}
        fig, axs = plt.subplots(2, 5, figsize=(19, 7.2), layout='constrained')
        if var == 'T':
            norm = mcolors.Normalize(185, 275); cm = 'viridis'
        else:
            L = np.nanpercentile(np.abs(np.concatenate([M[k][0].ravel() for k in M])), 99.5); norm = mcolors.Normalize(-L, L); cm = DIV
        for j, (k, nm) in enumerate(SRC):
            x, lat = M[k]
            im = axs[0, j].pcolormesh(lat, P, x, cmap=cm, norm=norm, shading='nearest', rasterized=True)
            axs[0, j].contour(lat, P, x, levels=np.arange(-100, 101, 10) if var == 'u' else np.arange(180, 300, 10), colors='k', linewidths=0.4)
            axp(axs[0, j]); axs[0, j].set_title(nm)
        fig.colorbar(im, ax=axs[0, :], shrink=0.85, extend='both').set_label(f'{var}, {unit}')
        D_ = {k: M['emu'][0] - on_emu_lat(*M[k]) for k, _ in SRC[1:]}
        Ld = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in D_.values()])), 99)
        axs[1, 0].axis('off')
        axs[1, 0].text(0.0, 0.5, f'{s} mean of 1997 and 1998\n(all {"Dec, Jan, Feb" if s == "DJF" else "Jun, Jul, Aug"} days\nof both years)\n\n'
                       'emulator, hlcorpus, c1, ERA5:\n06 UTC daily samples\nCESM: daily means\n\n'
                       'Bottom row: emulator minus\nreference, on the emulator\nlatitudes; numbers are\ncos-lat weighted over\n1-300 hPa', fontsize=8.5, va='center')
        for j, (k, nm) in enumerate(SRC[1:], start=1):
            r, m = wrms(D_[k]); stats.append(f'{var} {s} emulator - {k:5s}: RMS {r:6.2f} {unit.split("$")[0].strip()}  mean {m:+6.2f}')
            im2 = axs[1, j].pcolormesh(LE, P, D_[k], cmap=DIV, norm=mcolors.Normalize(-Ld, Ld), shading='nearest', rasterized=True)
            axp(axs[1, j]); axs[1, j].set_title(f'emulator minus {k}\nRMS {r:.2f}, mean {m:+.2f}')
        fig.colorbar(im2, ax=axs[1, :], shrink=0.85, extend='both').set_label(f'{var} difference, {unit}')
        fig.suptitle(f'Zonal-mean {var}, {s} 1997-1998, 1-300 hPa', fontsize=11)
        fig.savefig(f'{F}/climate2yr_{var}_{s}.png', dpi=140); plt.close(fig); print('wrote', f'climate2yr_{var}_{s}.png')
open(f'{D}/climate2yr_stats.txt', 'w').write('\n'.join(stats) + '\n'); print('\n'.join(stats))
