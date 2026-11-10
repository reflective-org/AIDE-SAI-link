"""Presentation version of fig_box_combined_6panel: Reflective brand colours and Open Sans, larger fonts,
x axis shared down each column, no figure title (the bin-exclusion rule and surface-area definition belong in
the speaker notes / caption). Same data, exclusion rule and box/whisker convention as plot_box_combined.py.
usage: plot_box_presentation.py cells.npz box_routed_merged.npz outdir [fontdir]
writes fig_box_6panel_presentation.png and .pdf"""
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

cells_f, box_f, outd = sys.argv[1:4]
for f in glob.glob(f'{sys.argv[4]}/*.ttf') if len(sys.argv) > 4 else []:
    font_manager.fontManager.addfont(f)

# Reflective brand palette (Brand Guidelines 2026)
YELLOW, GOLD, CLAY, CHARCOAL, SLATE, SAGE = '#FFBB00', '#A47F52', '#A0503C', '#282C2E', '#8B8D8E', '#BBC2BD'
RD = 287.05; NOCC = 1e-4; FLOOR = 1e-3

c = np.load(cells_f); b = np.load(box_f)
dp = c['dp_bin_nm']
T = np.concatenate([c['bg_T'], c['sai_T']]); P = np.concatenate([c['bg_P'], c['sai_P']]); rho = P / (RD * T)

plt.rcParams.update({
    'font.family': 'Open Sans', 'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 17,
    'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 16,
    'text.color': CHARCOAL, 'axes.labelcolor': CHARCOAL, 'axes.edgecolor': SLATE,
    'xtick.color': SLATE, 'ytick.color': SLATE, 'xtick.labelcolor': CHARCOAL, 'ytick.labelcolor': CHARCOAL,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 1.0,
})


def get(eng, q, t):
    """per-bin N, M, or dry surface area S (proportional to N^(1/3) M^(2/3); constants cancel in relative error)."""
    if q != 'S': return b[f'{eng}_{q}'][t]
    return np.cbrt(np.maximum(b[f'{eng}_N'][t], 0)) * np.maximum(b[f'{eng}_M'][t], 0) ** (2 / 3)


def boxes(ax, data, pos, width):
    bp = ax.boxplot(data, positions=pos, whis=(5, 95), showfliers=False, widths=width, patch_artist=True,
                    medianprops=dict(color=CHARCOAL, lw=1.6))
    for p_ in bp['boxes']: p_.set(facecolor=YELLOW, alpha=0.75, edgecolor=GOLD, lw=1.2)
    for w in bp['whiskers'] + bp['caps']: w.set(color=GOLD, lw=1.4)



fig, axs = plt.subplots(3, 2, figsize=(16, 10.5), sharex='col', gridspec_kw=dict(width_ratios=[1, 2.6]))
hours = [6, 12, 18, 24]; xmax = 0
for i, (q, qn) in enumerate((('N', 'Number'), ('M', 'Mass'), ('S', 'Surface area'))):
    ax = axs[i, 0]; data = []
    for t in range(1, 5):
        n = get('fast', q, t).sum(1); e = get('emulator', q, t).sum(1); data.append((e - n) / n * 100)
    boxes(ax, data, hours, 3)
    ax.plot(hours, [np.median(np.abs(x)) for x in data], '--o', color=CLAY, lw=1.8, ms=5)
    ax.axhline(0, color=SAGE, lw=1, zorder=0)
    ax.set_title(f'{qn}: total per cell', loc='left', fontweight='semibold')

    ax = axs[i, 1]
    n = get('fast', q, 4); e = get('emulator', q, 4); mx = n.max(1, keepdims=True)
    ok = (n >= FLOOR * mx) & (mx > 0) & (b['fast_N'][4] * rho[:, None] >= NOCC)
    r = np.where(ok, (e - np.where(ok, n, 1)) / np.where(ok, n, 1), np.nan) * 100
    data = [x[np.isfinite(x)] for x in r.T]; keep = [k for k in range(40) if data[k].size >= 20]
    xmax = max(xmax, max(keep))
    boxes(ax, [data[k] for k in keep], keep, 0.6)
    ax.plot(range(40), [np.median(np.abs(data[k])) if k in keep else np.nan for k in range(40)],
            '--o', color=CLAY, lw=1.8, ms=3.5)
    ax.axhline(0, color=SAGE, lw=1, zorder=0)
    ax.set_title(f'{qn}: per size bin at 24 h', loc='left', fontweight='semibold')
    print(f'{q}: bin0 median |err| {np.median(np.abs(data[0])):.2f} %, total 24 h median |err| '
          f'{np.median(np.abs((e.sum(1) - n.sum(1)) / n.sum(1) * 100)):.3f} %')

axs[2, 0].set_xticks(hours); axs[2, 0].set_xlim(2, 28); axs[2, 0].set_xlabel('Hours into the box run')
# bins are log-uniform in diameter, so a round diameter d sits at fractional bin index interp(log d, log dp, k)
k_of = lambda d: np.interp(np.log(d), np.log(dp), np.arange(dp.size))
round_d = [d for d in (2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000) if dp[0] <= d <= dp[xmax]]
axs[2, 1].set_xticks([k_of(d) for d in round_d]); axs[2, 1].set_xticklabels([f'{d:,}' for d in round_d])
axs[2, 1].set_xticks(range(xmax + 1), minor=True); axs[2, 1].tick_params(axis='x', which='minor', length=3)
axs[2, 1].set_xlim(-1, xmax + 1); axs[2, 1].set_xlabel('Particle dry diameter (nm)')
fig.supylabel('Relative error, emulator vs numerical (%)', fontsize=17, color=CHARCOAL)

handles = [Patch(facecolor=YELLOW, alpha=0.75, edgecolor=GOLD, label='25–75 % of cells'),
           Line2D([], [], color=GOLD, lw=1.4, label='5–95 %'),
           Line2D([], [], color=CHARCOAL, lw=1.6, label='Median'),
           Line2D([], [], color=CLAY, lw=1.8, ls='--', marker='o', ms=5, label='Median |error|')]
fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.2)
for ext in ('png', 'pdf'):
    fig.savefig(f'{outd}/fig_box_6panel_presentation.{ext}', dpi=200, facecolor='white')
print('wrote fig_box_6panel_presentation.png/.pdf')
