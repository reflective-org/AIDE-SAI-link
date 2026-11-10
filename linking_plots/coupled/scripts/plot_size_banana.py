"""Banana plots: stratospheric column size distribution vs time, global / NH / SH, for TOMAS numerical, the TOMAS
emulator and MAM4 (injection runs, 10 Tg SO2/yr, 2-year coupled campaign). One figure for number (dN/dlogD) and one
for dry surface area (dS/dlogD). Reflective brand styling, presentation fonts.
Reads cache/size_ts_v2.npz from size_ts.py (one frame per day at 12 UTC).
Surface area per bin: S_k = N_k * pi * D_k^2, D_k = dry diameter of the bin's column-mean particle mass (rho 1770).
Also writes *_15-45 versions with the 15-45N and 15-45S bands.
usage: plot_size_banana.py outdir [fontdir]"""
import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm

outd = sys.argv[1]; os.makedirs(outd, exist_ok=True)
for f in glob.glob(f'{sys.argv[2]}/*.ttf') if len(sys.argv) > 2 else []:
    font_manager.fontManager.addfont(f)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache')
Z = np.load(f'{CACHE}/size_ts_v2.npz')

CHARCOAL, SLATE = '#282C2E', '#8B8D8E'
# sequential map from the brand palette: Sand -> Yellow -> Red -> Purple -> Charcoal
CMAP = LinearSegmentedColormap.from_list('reflective_seq', ['#F6EFEA', '#FFBB00', '#D46448', '#655886', '#282C2E'])
CMAP.set_under('white')
plt.rcParams.update({
    'font.family': 'Open Sans', 'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 17,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'text.color': CHARCOAL, 'axes.labelcolor': CHARCOAL, 'axes.edgecolor': SLATE,
    'xtick.color': SLATE, 'ytick.color': SLATE, 'xtick.labelcolor': CHARCOAL, 'ytick.labelcolor': CHARCOAL,
})

RHO = 1770.0
dp = Z['dp_bin_nm']; nb = dp.size
dlog = np.log10(dp[1] / dp[0])                                         # bins are log-uniform
edges = np.concatenate([dp * 10 ** (-dlog / 2), [dp[-1] * 10 ** (dlog / 2)]])
day = Z['day']
t = np.datetime64('1997-01-01T12') + day.astype('timedelta64[D]')     # 365-day years; 1997-98 have no leap day
t_edges = np.concatenate([t - np.timedelta64(12, 'h'), [t[-1] + np.timedelta64(12, 'h')]])

SCHEMES = (('num', 'TOMAS numerical'), ('emu', 'TOMAS emulator'), ('mam', 'MAM4'))
HEMI = (('glob', 'Global'), ('nh', 'Northern Hemisphere'), ('sh', 'Southern Hemisphere'))
BANDS = (('n1545', '15–45°N'), ('s1545', '15–45°S'))


def field(key, reg, q):
    N = Z[f'{key}_N_{reg}']; M = Z[f'{key}_M_{reg}']                  # (ndays, nb): # m-2, kg m-2
    if q == 'N':
        return N / dlog                                               # # m-2 per log10 D
    D = np.where(N > 0, np.cbrt(np.maximum(M, 0) / np.maximum(N, 1e-300) / RHO * 6 / np.pi), dp * 1e-9)
    return N * np.pi * D ** 2 / dlog                                  # m2 of particle surface per m2 of ground, per log10 D


def figure(q, fname, cblabel, decades, REGIONS):
    F = {(k, r): field(k, r, q) for k, _ in SCHEMES for r, _ in REGIONS}
    vmax = max(np.percentile(v, 99.9) for v in F.values()); norm = LogNorm(vmax / 10 ** decades, vmax)
    nr = len(REGIONS)
    fig, axs = plt.subplots(nr, 3, figsize=(17, 3.5 * nr), sharex=True, sharey=True, squeeze=False)
    for j, (k, kn) in enumerate(SCHEMES):
        for i, (r, rn) in enumerate(REGIONS):
            ax = axs[i, j]
            pm = ax.pcolormesh(t_edges, edges, F[(k, r)].T, cmap=CMAP, norm=norm, shading='flat', rasterized=True)
            ax.set_yscale('log'); ax.set_ylim(2, 5000)
            ax.set_yticks([2, 10, 100, 1000]); ax.set_yticklabels(['2', '10', '100', '1,000'])
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            if i == 0: ax.set_title(kn, fontweight='semibold')
            if j == 0: ax.set_ylabel(rn, fontweight='semibold')
            for s in ax.spines.values(): s.set_color(SLATE)
    fig.supylabel('Particle dry diameter (nm)', fontsize=17, color=CHARCOAL, x=0.005)
    fig.tight_layout(rect=(0.015, 0, 0.9, 1), h_pad=0.8, w_pad=0.8)
    cax = fig.add_axes([0.915, 0.12 if nr == 3 else 0.16, 0.016, 0.76 if nr == 3 else 0.70])
    cb = fig.colorbar(pm, cax=cax, extend='min'); cb.set_label(cblabel); cb.outline.set_edgecolor(SLATE)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{outd}/{fname}.{ext}', dpi=200, facecolor='white')
    plt.close(fig); print('wrote', f'{outd}/{fname}.png/.pdf', f'(colour range {norm.vmin:.3g} - {norm.vmax:.3g})')


NL = 'Particles per m$^2$ of ground, dN/dlog$_{10}$D'
SL = 'Particle surface per m$^2$ of ground,\ndS/dlog$_{10}$D (m$^2$ m$^{-2}$)'
for regs, sfx in ((HEMI, ''), (BANDS, '_15-45')):
    figure('N', f'size_distribution_banana_number{sfx}', NL, 4, regs)
    figure('S', f'size_distribution_banana_surface_area{sfx}', SL, 3, regs)
