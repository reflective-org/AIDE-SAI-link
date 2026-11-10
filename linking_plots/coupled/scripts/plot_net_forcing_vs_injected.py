"""Net TOA SAI forcing vs cumulative SO2 injected, 2-year coupled run (emu_campaign_1997 + _y2, 730 days),
TOMAS emulator / TOMAS numerical / MAM4. Reflective brand styling, presentation fonts.

Reads cache/timeseries_v2.npz: Fnet_raw = (SW_up + OLR)_control - (SW_up + OLR)_injection, area-weighted global
mean of the raw six-hourly frames (exact for the global mean), W m-2; negative = cooling.
  (a) net forcing (thin: daily, thick: 30-day running mean)
  (b) net forcing per Tg of SO2 injected so far, from the 30-day running mean (first 30 days dropped)
x = cumulative SO2 injected, 10 Tg SO2/yr continuous from day 1.
usage: plot_net_forcing_vs_injected.py outdir [fontdir]"""
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

YELLOW, BLUE, CHARCOAL, SLATE, SAGE = '#FFBB00', '#466F8D', '#282C2E', '#8B8D8E', '#BBC2BD'
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

INJ = 10.0; SKIP = 30
nd = TS['num_Fnet_raw'].size; day = np.arange(1, nd + 1)
x = INJ * (day - 0.5) / 365.0                                 # cumulative Tg SO2 at the middle of each day


def run30(v):
    k = np.ones(30) / 30
    return np.convolve(v, k, 'same') / np.convolve(np.ones_like(v), k, 'same')


fig, axs = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
res = {}
for k in ('mam', 'num', 'emu'):
    F = TS[f'{k}_Fnet_raw']; Fs = run30(F)
    per = Fs / x; per[:SKIP] = np.nan
    res[k] = (Fs, per)
    axs[0].plot(x, F, color=STYLE[k]['color'], lw=0.7, alpha=0.35)
    axs[0].plot(x, Fs, **STYLE[k]); axs[1].plot(x, per, **STYLE[k])
axs[0].set_ylabel('W m$^{-2}$'); axs[0].set_title('Net forcing at the top of the atmosphere', loc='left', fontweight='semibold')
axs[1].set_ylabel('W m$^{-2}$ per Tg SO$_2$'); axs[1].set_title('Net forcing per Tg of SO$_2$ injected so far', loc='left', fontweight='semibold')
axs[1].set_xlabel('Cumulative SO$_2$ injected (Tg)'); axs[1].set_xlim(0, x[-1] + 0.05); axs[1].set_xticks(range(0, 21, 2))
for ax in axs:
    ax.set_ylim(top=0); ax.axhline(0, color=SLATE, lw=1); ax.grid(axis='y', color=SAGE, lw=0.6, alpha=0.6)
top = axs[0].secondary_xaxis('top', functions=(lambda v: v / INJ, lambda v: v * INJ))
top.set_xticks([0, 0.5, 1, 1.5, 2]); top.set_xlabel('Years of injection (10 Tg SO$_2$ per year)')
h, l = axs[0].get_legend_handles_labels(); order = [l.index(STYLE[k]['label']) for k in ('num', 'emu', 'mam')]
fig.legend([h[i] for i in order], [l[i] for i in order], loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.5)
for ext in ('png', 'pdf'):
    fig.savefig(f'{outd}/net_forcing_vs_injected.{ext}', dpi=200, facecolor='white')

for k in ('num', 'emu', 'mam'):
    Fs, per = res[k]
    print(k, '  '.join(f'{x[d-1]:4.1f} Tg: {Fs[d-1]:+.2f} W/m2 ({per[d-1]:+.3f}/Tg)' for d in (91, 182, 365, 545, 700)),
          f'| min 30d {Fs.min():+.2f} at {x[Fs.argmin()]:.1f} Tg')
print('wrote', f'{outd}/net_forcing_vs_injected.png/.pdf')
