"""Scattering efficiency vs cumulative SO2 injected, 2-year coupled run (emu_campaign_1997 + _y2, 730 days),
TOMAS emulator / TOMAS numerical / MAM4. Reflective brand styling, presentation fonts.

Reads cache/timeseries_v2.npz (daily, area-weighted global means from agg2.py):
  dAOD      : AOD550 injection minus control (total column)
  dM        : stratospheric column sulfate, injection minus control, mg m-2 (tropopause-filtered)
  Fsw_raw   : SW TOA forcing, control minus injection SW_up, W m-2 (global mean of raw frames is exact)
Plotted (30-day running means of numerator and denominator, then the ratio; first 30 days dropped):
  (a) mass extinction efficiency  dAOD / dM            [m2 g-1]  (sulfate at 550 nm barely absorbs, so ~ scattering)
  (b) SW forcing per unit burden  -Fsw / burden        [W m-2 per Tg]
x = cumulative SO2 injected, 10 Tg SO2/yr continuous from day 1.
usage: plot_scattering_efficiency.py outdir [fontdir]"""
import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

outd = sys.argv[1]; os.makedirs(outd, exist_ok=True)
for f in glob.glob(f'{sys.argv[2]}/*.ttf') if len(sys.argv) > 2 else []:
    font_manager.fontManager.addfont(f)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache')
TS = np.load(f'{CACHE}/timeseries_v2.npz')

YELLOW, GOLD, BLUE, CHARCOAL, SLATE, SAGE = '#FFBB00', '#A47F52', '#466F8D', '#282C2E', '#8B8D8E', '#BBC2BD'
STYLE = {'num': dict(color=CHARCOAL, lw=2.6, ls='-', label='TOMAS numerical (reference)'),
         'emu': dict(color=YELLOW, lw=3.0, ls=(0, (4, 2)), label='TOMAS emulator'),
         'mam': dict(color=BLUE, lw=2.6, ls='-', label='MAM4')}
plt.rcParams.update({
    'font.family': 'Open Sans', 'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 17,
    'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 16,
    'text.color': CHARCOAL, 'axes.labelcolor': CHARCOAL, 'axes.edgecolor': SLATE,
    'xtick.color': SLATE, 'ytick.color': SLATE, 'xtick.labelcolor': CHARCOAL, 'ytick.labelcolor': CHARCOAL,
    'axes.spines.top': False, 'axes.spines.right': False,
})

INJ = 10.0; AREA = 5.1007e14; SKIP = 30                      # Tg SO2/yr; Earth area m2; spin-up days dropped
nd = TS['num_dM'].size; day = np.arange(1, nd + 1)
x = INJ * (day - 0.5) / 365.0                                 # cumulative Tg SO2 at the middle of each day


def run30(v):
    k = np.ones(30) / 30
    return np.convolve(v, k, 'same') / np.convolve(np.ones_like(v), k, 'same')


eff = {}
for k in ('num', 'emu', 'mam'):
    dM = run30(TS[f'{k}_dM']); burden_tg = dM * AREA * 1e-15            # mg m-2 -> Tg
    mee = run30(TS[f'{k}_dAOD']) / (dM * 1e-3)                          # m2 g-1
    fpb = -run30(TS[f'{k}_Fsw_raw']) / burden_tg                         # W m-2 per Tg
    mee[:SKIP] = fpb[:SKIP] = np.nan
    eff[k] = (mee, fpb, burden_tg)

fig, axs = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
for k in ('mam', 'num', 'emu'):
    mee, fpb, _ = eff[k]
    axs[0].plot(x, mee, **STYLE[k]); axs[1].plot(x, fpb, **STYLE[k])
axs[0].set_ylabel('m$^2$ per g of sulfate'); axs[0].set_title('AOD (550 nm) per unit stratospheric sulfate', loc='left', fontweight='semibold')
axs[1].set_ylabel('W m$^{-2}$ per Tg sulfate'); axs[1].set_title('Shortwave cooling per unit stratospheric sulfate', loc='left', fontweight='semibold')
axs[1].set_xlabel('Cumulative SO$_2$ injected (Tg)'); axs[1].set_xlim(0, x[-1] + 0.05); axs[1].set_xticks(range(0, 21, 2))
for ax in axs:
    ax.set_ylim(bottom=0); ax.grid(axis='y', color=SAGE, lw=0.6, alpha=0.6)
top = axs[0].secondary_xaxis('top', functions=(lambda v: v / INJ, lambda v: v * INJ))
top.set_xticks([0, 0.5, 1, 1.5, 2]); top.set_xlabel('Years of injection (10 Tg SO$_2$ per year)')
top.spines['top'].set_visible(True); top.spines['top'].set_color(SLATE)
h, l = axs[0].get_legend_handles_labels(); order = [l.index(STYLE[k]['label']) for k in ('num', 'emu', 'mam')]
fig.legend([h[i] for i in order], [l[i] for i in order], loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.5)
for ext in ('png', 'pdf'):
    fig.savefig(f'{outd}/scattering_efficiency_vs_injected.{ext}', dpi=200, facecolor='white')

for k in ('num', 'emu', 'mam'):
    mee, fpb, b = eff[k]
    for d in (60, 182, 365, 545, nd):
        i = d - 1
        print(f'{k} day {d:3d} ({x[i]:5.2f} Tg SO2): burden {b[i]:.2f} Tg, MEE {mee[i]:.2f} m2/g, SW/burden {fpb[i]:.3f} W m-2/Tg')
print('wrote', f'{outd}/scattering_efficiency_vs_injected.png/.pdf')
