"""Combined (background + SAI pooled, 10,000 cells) versions of the box-test figures, no persistence line.
Same data, bin-exclusion rule and box/whisker convention as plot_box.py.
usage: plot_box_combined.py cells.npz box.npz outdir [tag] [label]
writes  fig3c_totals_vs_time_combined{tag}.png, fig1c_perbin_number_combined{tag}.png, fig1c_perbin_mass_combined{tag}.png"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cells_f, box_f, outd = sys.argv[1:4]
tag = sys.argv[4] if len(sys.argv) > 4 else ''
LAB = sys.argv[5] if len(sys.argv) > 5 else 'Deployed TOMAS emulator'
CE = '#eb6834'; RD = 287.05; NOCC = 1e-4; FLOOR = 1e-3
c = np.load(cells_f); b = np.load(box_f)
dp = c['dp_bin_nm']; ncell = c['bg_num'].shape[0] + c['sai_num'].shape[0]
T = np.concatenate([c['bg_T'], c['sai_T']]); P = np.concatenate([c['bg_P'], c['sai_P']]); rho = P / (RD * T)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def style(bp):
    for p_ in bp['boxes']: p_.set(facecolor=CE, alpha=0.55, edgecolor=CE)
    for w in bp['whiskers'] + bp['caps']: w.set(color=CE)


# ---- totals vs time, pooled ----
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, (q, qn) in zip(axs, (('N', 'total number'), ('M', 'total mass'))):
    data = []
    for t in range(1, 5):
        n = b[f'fast_{q}'][t].sum(1); e = b[f'emulator_{q}'][t].sum(1); data.append((e - n) / n * 100)
    bp = ax.boxplot(data, positions=[6, 12, 18, 24], whis=(5, 95), showfliers=False, widths=3, patch_artist=True,
                    medianprops=dict(color='k', lw=1)); style(bp)
    med = [np.median(np.abs(x)) for x in data]
    ax.plot([6, 12, 18, 24], med, 'k--.', lw=1, label='median |rel. err|')
    ax.axhline(0, color='0.6', lw=0.6); ax.set_xticks([6, 12, 18, 24]); ax.set_xlim(2, 28)
    ax.set_xlabel('hours into the box run'); ax.set_title(f'{qn} per cell', loc='left')
    for t, x in zip((6, 12, 18, 24), data):
        print(f'tot{q} {t:2d} h: median |err| {np.median(np.abs(x)):.4f} %  p90 {np.percentile(np.abs(x), 90):.4f} %  '
              f'5-95 % {np.percentile(x, 5):+.3f} .. {np.percentile(x, 95):+.3f} %')
axs[0].set_ylabel('relative error, emulator vs numerical [%]'); axs[0].legend(frameon=False)
fig.suptitle(f'{LAB} vs TOMAS numerical, all {ncell:,} cells (background + SAI plume): per-cell total error\n'
             'boxes 25-75 %, whiskers 5-95 %', fontsize=10)
fig.tight_layout(); fig.savefig(f'{outd}/fig3c_totals_vs_time_combined{tag}.png', dpi=150); plt.close(fig)

# ---- per-bin, pooled, 24 h ----
for q, qn in (('N', 'number'), ('M', 'mass')):
    n = b[f'fast_{q}'][4]; e = b[f'emulator_{q}'][4]; mx = n.max(1, keepdims=True)
    ok = (n >= FLOOR * mx) & (mx > 0) & (b['fast_N'][4] * rho[:, None] >= NOCC)
    r = np.where(ok, (e - np.where(ok, n, 1)) / np.where(ok, n, 1), np.nan) * 100
    data = [x[np.isfinite(x)] for x in r.T]; keep = [k for k in range(40) if data[k].size >= 20]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    bp = ax.boxplot([data[k] for k in keep], positions=keep, whis=(5, 95), showfliers=False, widths=0.6,
                    patch_artist=True, medianprops=dict(color='k', lw=1)); style(bp)
    med = np.array([np.median(np.abs(data[k])) if k in keep else np.nan for k in range(40)])
    ax.plot(range(40), med, 'k--.', lw=1.2, ms=3, label='median |rel. err|')
    ax.axhline(0, color='0.6', lw=0.6); ax.set_yscale('symlog', linthresh=10, linscale=1.0)
    ax.set_xticks(range(0, 40, 3)); ax.set_xticklabels([f'{dp[k]:.3g}' for k in range(0, 40, 3)], rotation=45)
    ax.set_xlim(-1, 40); ax.set_xlabel('bin dry diameter [nm]'); ax.set_ylabel('relative error, emulator vs numerical [%]')
    ax.legend(loc='upper right', frameon=False)
    ax.set_title(f'{LAB} vs TOMAS numerical: per-bin {qn} at 24 h, all {ncell:,} cells (background + SAI plume)\n'
                 f'boxes 25-75 %, whiskers 5-95 %; bins excluded where numerical < {FLOOR:g} x the cell\'s largest bin '
                 f'or < {NOCC:g} particles per m$^3$', fontsize=9.5, loc='left')
    fig.tight_layout(); fig.savefig(f'{outd}/fig1c_perbin_{qn}_combined{tag}.png', dpi=150); plt.close(fig)
    print(f'bin{q} 24 h: all included bins median |err| {np.nanmedian(np.abs(r)):.3f} %  p90 {np.nanpercentile(np.abs(r), 90):.3f} %  '
          f'included {np.isfinite(r).mean():.3f} of (cell, bin) pairs; bin0 median |err| {med[0]:.2f} %')

# ---- all four panels in one figure: rows number / mass, columns totals vs time | per-bin at 24 h ----
def get(eng, q, t):
    """per-bin N, M, or dry surface area S. S_k = N_k pi D_k^2 with D_k = (6 M_k / (pi rho N_k))^(1/3), so
    S_k is proportional to N_k^(1/3) M_k^(2/3); the constant (density, pi) cancels in every relative error."""
    if q != 'S': return b[f'{eng}_{q}'][t]
    return np.cbrt(np.maximum(b[f'{eng}_N'][t], 0)) * np.maximum(b[f'{eng}_M'][t], 0) ** (2 / 3)


fig, axs = plt.subplots(3, 2, figsize=(16, 12), gridspec_kw=dict(width_ratios=[1, 2.7]))
for i, (q, qn) in enumerate((('N', 'number'), ('M', 'mass'), ('S', 'surface area'))):
    ax = axs[i, 0]; data = []
    for t in range(1, 5):
        n = get('fast', q, t).sum(1); e = get('emulator', q, t).sum(1); data.append((e - n) / n * 100)
    style(ax.boxplot(data, positions=[6, 12, 18, 24], whis=(5, 95), showfliers=False, widths=3, patch_artist=True,
                     medianprops=dict(color='k', lw=1)))
    ax.plot([6, 12, 18, 24], [np.median(np.abs(x)) for x in data], 'k--.', lw=1, label='median |rel. err|')
    ax.axhline(0, color='0.6', lw=0.6); ax.set_xticks([6, 12, 18, 24]); ax.set_xlim(2, 28)
    ax.set_title(f'({"ace"[i]}) total {qn} per cell vs time', loc='left'); ax.set_ylabel('relative error, emulator vs numerical [%]')
    if i == 2: ax.set_xlabel('hours into the box run')
    if i == 0: ax.legend(frameon=False, loc='lower left')
    ax = axs[i, 1]
    n = get('fast', q, 4); e = get('emulator', q, 4); mx = n.max(1, keepdims=True)
    ok = (n >= FLOOR * mx) & (mx > 0) & (b['fast_N'][4] * rho[:, None] >= NOCC)
    r = np.where(ok, (e - np.where(ok, n, 1)) / np.where(ok, n, 1), np.nan) * 100
    data = [x[np.isfinite(x)] for x in r.T]; keep = [k for k in range(40) if data[k].size >= 20]
    print(f'6panel {q}: total 24 h median |err| {np.median(np.abs(data_t := ((get("emulator", q, 4).sum(1) - get("fast", q, 4).sum(1)) / get("fast", q, 4).sum(1) * 100))):.3f} %, p90 {np.percentile(np.abs(data_t), 90):.3f} %; '
          f'per bin median |err| {np.nanmedian(np.abs(r)):.3f} %, p90 {np.nanpercentile(np.abs(r), 90):.3f} %, bin0 median {np.median(np.abs(data[0])) if data[0].size else float("nan"):.2f} %')
    style(ax.boxplot([data[k] for k in keep], positions=keep, whis=(5, 95), showfliers=False, widths=0.6,
                     patch_artist=True, medianprops=dict(color='k', lw=1)))
    ax.plot(range(40), [np.median(np.abs(data[k])) if k in keep else np.nan for k in range(40)], 'k--.', lw=1.2, ms=3)
    ax.axhline(0, color='0.6', lw=0.6)    # linear y axis (user request)
    ax.set_xticks(range(0, 40, 3)); ax.set_xticklabels([f'{dp[k]:.3g}' for k in range(0, 40, 3)], rotation=45); ax.set_xlim(-1, 40)
    ax.set_title(f'({"bdf"[i]}) {qn} per bin at 24 h', loc='left')
    if i == 2: ax.set_xlabel('bin dry diameter [nm]')
fig.suptitle(f'{LAB} vs TOMAS numerical, 24 h box runs, all {ncell:,} cells (background + SAI plume pooled)\n'
             f'boxes 25-75 %, whiskers 5-95 %; per-bin panels exclude bins where numerical < {FLOOR:g} x the cell\'s largest bin '
             f'or < {NOCC:g} particles per m$^3$; surface area = dry, from each bin\'s mean particle mass (∝ N$^{{1/3}}$M$^{{2/3}}$)', fontsize=10)
fig.tight_layout(); fig.savefig(f'{outd}/fig_box_combined_6panel{tag}.png', dpi=150); plt.close(fig)
print('wrote fig_box_combined_6panel')
