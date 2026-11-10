"""Figures + summary numbers for the 10k-cell 24 h box comparison.
usage: plot_box.py cells.npz box.npz outdir [tag]"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cells_f, box_f, outd = sys.argv[1:4]
tag = sys.argv[4] if len(sys.argv) > 4 else ''
LAB = sys.argv[5] if len(sys.argv) > 5 else 'TOMAS emulator'
CN, CE = '#2a78d6', '#eb6834'
RD = 287.05
c = np.load(cells_f); b = np.load(box_f)
dp = c['dp_bin_nm']
nbg = c['bg_num'].shape[0]; N = nbg + c['sai_num'].shape[0]
pop = {'Background': slice(0, nbg), 'SAI plume': slice(nbg, N)}
T = np.concatenate([c['bg_T'], c['sai_T']]); P = np.concatenate([c['bg_P'], c['sai_P']])
rho = P / (RD * T)                                    # kg/m3
NOCC = 1e-4                                           # 10x the TOMAS Nk floor (1e-5 #/box, box = 1 m3): floor-level bins are 'empty'
FLOOR = 1e-3                                          # bin included if num >= FLOOR x cell max bin (numerical)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def rel(e, n):
    return (e - n) / n


def binerr(q, t=4):
    n = b[f'fast_{q}'][t]; e = b[f'emulator_{q}'][t]
    mx = n.max(1, keepdims=True)
    nbox = b[f'fast_N'][t] * rho[:, None]               # #/box (1 m3)
    ok = (n >= FLOOR * mx) & (mx > 0) & (nbox >= NOCC)
    r = np.where(ok, rel(e, np.where(ok, n, 1.0)), np.nan)
    return r


summ = {}
# ---------------- Fig 1: per-bin box plots ----------------
fig, axs = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
for i, (q, qn) in enumerate((('N', 'number'), ('M', 'mass'))):
    R = binerr(q)
    for j, (pn, sl) in enumerate(pop.items()):
        ax = axs[i, j]; r = R[sl] * 100
        data = [x[np.isfinite(x)] for x in r.T]
        pos = np.arange(40)
        keep = [k for k in range(40) if data[k].size >= 20]
        bp = ax.boxplot([data[k] for k in keep], positions=pos[keep], whis=(5, 95), showfliers=False,
                        widths=0.6, patch_artist=True, medianprops=dict(color='k', lw=1))
        for p_ in bp['boxes']:
            p_.set(facecolor=CE, alpha=0.55, edgecolor=CE)
        for w in bp['whiskers'] + bp['caps']:
            w.set(color=CE)
        med = np.array([np.median(np.abs(data[k])) if k in keep else np.nan for k in range(40)])
        ax.plot(pos, med, color='k', lw=1.2, ls='--', marker='.', ms=3, label='median |rel. err|')
        n0 = b[f'fast_{q}'][0][sl]; n4 = b[f'fast_{q}'][4][sl]
        pc = np.where(np.isfinite(r), np.abs(n4 / np.where(n0 > 0, n0, np.nan) - 1) * 100, np.nan)
        pmed = np.array([np.nanmedian(pc[:, k]) if k in keep else np.nan for k in range(40)])
        ax.plot(pos, pmed, color='0.55', lw=1, ls=':', label='median |numerical 24 h change| (persistence error)')
        ax.axhline(0, color='0.6', lw=0.6)
        ax.set_yscale('symlog', linthresh=10, linscale=1.0)
        ax.set_title(f'{pn}: per-bin {qn}, 24 h', loc='left')
        if j == 0:
            ax.set_ylabel('relative error, emulator vs numerical [%]')
        frac = np.array([data[k].size for k in range(40)]) / r.shape[0]
        summ[f'{pn}|bin_{q}'] = dict(median_abs=[float(x) for x in med],
                                     p90_abs=[float(np.percentile(np.abs(data[k]), 90)) if k in keep else None for k in range(40)],
                                     n_included=[int(data[k].size) for k in range(40)],
                                     all_bins_median_abs=float(np.nanmedian(np.abs(r))),
                                     all_bins_p90_abs=float(np.nanpercentile(np.abs(r), 90)),
                                     persistence_all_bins_median_abs=float(np.nanmedian(pc)),
                                     persistence_all_bins_p90_abs=float(np.nanpercentile(pc, 90)),
                                     frac_bins_included=float(np.isfinite(r).mean()))
        if i == 0 and j == 0:
            ax.legend(loc='upper left', frameon=False)
for ax in axs[1]:
    ax.set_xticks(np.arange(0, 40, 3))
    ax.set_xticklabels([f'{dp[k]:.3g}' for k in range(0, 40, 3)], rotation=45)
    ax.set_xlabel('bin dry diameter [nm]')
fig.suptitle(f'{LAB} vs TOMAS numerical, 24 h box runs (5,000 background + 5,000 SAI cells)\n'
             f'boxes 25-75 %, whiskers 5-95 %; bins excluded where numerical < {FLOOR:g} x cell max bin or < {NOCC:g} particles per m$^3$', fontsize=9)
fig.tight_layout(); fig.savefig(f'{outd}/fig1_perbin_relerr{tag}.png', dpi=150); plt.close(fig)

# ---------------- Fig 2: mean/median distributions ----------------
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
for i, (q, qn, conv, unit) in enumerate((('N', 'number', 1e-6, 'dN per bin [cm$^{-3}$]'),
                                         ('M', 'mass', 1e9, 'dM per bin [µg SO$_4$ m$^{-3}$]'))):
    for j, (pn, sl) in enumerate(pop.items()):
        ax = axs[i, j]
        f = lambda a: a[sl] * rho[sl, None] * conv
        n0 = f(b[f'fast_{q}'][0]); n24 = f(b[f'fast_{q}'][4]); e24 = f(b[f'emulator_{q}'][4])
        ax.plot(dp, n0.mean(0), color='0.65', ls='--', lw=1, label='t = 0 mean')
        ax.plot(dp, n24.mean(0), color=CN, lw=1.8, label='numerical, mean')
        ax.plot(dp, e24.mean(0), color=CE, lw=1.8, label='emulator, mean')
        ax.plot(dp, np.median(n24, 0), color=CN, lw=1.2, ls=':', label='numerical, median')
        ax.plot(dp, np.median(e24, 0), color=CE, lw=1.2, ls=':', label='emulator, median')
        ax.set_xscale('log'); ax.set_yscale('log')
        top = max(n24.mean(0).max(), n0.mean(0).max())
        ax.set_ylim(top * 1e-8, top * 3)
        ax.set_title(f'{pn}: {qn} distribution at 24 h', loc='left')
        if j == 0:
            ax.set_ylabel(unit)
        if i == 1:
            ax.set_xlabel('bin dry diameter [nm]')
        if i == 0 and j == 0:
            ax.legend(frameon=False, fontsize=8)
fig.suptitle(f'{LAB} vs TOMAS numerical, 24 h box runs', fontsize=9)
fig.tight_layout(); fig.savefig(f'{outd}/fig2_mean_distribution{tag}.png', dpi=150); plt.close(fig)

# ---------------- Fig 3: totals vs time ----------------
fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for i, (q, qn) in enumerate((('N', 'total number'), ('M', 'total mass'))):
    for j, (pn, sl) in enumerate(pop.items()):
        ax = axs[i, j]; data = []
        for t in range(1, 5):
            n = b[f'fast_{q}'][t][sl].sum(1); e = b[f'emulator_{q}'][t][sl].sum(1)
            r = rel(e, n) * 100; data.append(r)
            summ[f'{pn}|tot{q}|{6*t}h'] = dict(median=float(np.median(r)), median_abs=float(np.median(np.abs(r))),
                                               p90_abs=float(np.percentile(np.abs(r), 90)),
                                               p99_abs=float(np.percentile(np.abs(r), 99)),
                                               p05=float(np.percentile(r, 5)), p95=float(np.percentile(r, 95)))
        bp = ax.boxplot(data, positions=[6, 12, 18, 24], whis=(5, 95), showfliers=False, widths=3,
                        patch_artist=True, medianprops=dict(color='k', lw=1))
        for p_ in bp['boxes']:
            p_.set(facecolor=CE, alpha=0.55, edgecolor=CE)
        for w in bp['whiskers'] + bp['caps']:
            w.set(color=CE)
        ax.plot([6, 12, 18, 24], [np.median(np.abs(x)) for x in data], 'k--.', lw=1, label='median |rel. err|')
        pers = [np.median(np.abs(b[f'fast_{q}'][t][sl].sum(1) / b[f'fast_{q}'][0][sl].sum(1) - 1)) * 100 for t in range(1, 5)]
        ax.plot([6, 12, 18, 24], pers, color='0.55', ls=':', marker='.', lw=1, label='median |numerical change| (persistence)')
        for t in range(1, 5):
            summ[f'{pn}|tot{q}|{6*t}h']['persistence_median_abs'] = float(pers[t - 1])
            summ[f'{pn}|tot{q}|{6*t}h']['persistence_p90_abs'] = float(np.percentile(np.abs(b[f'fast_{q}'][t][sl].sum(1) / b[f'fast_{q}'][0][sl].sum(1) - 1), 90) * 100)
        ax.axhline(0, color='0.6', lw=0.6)
        ax.set_title(f'{pn}: {qn} per cell', loc='left')
        if j == 0:
            ax.set_ylabel('relative error, emulator vs numerical [%]')
        if i == 1:
            ax.set_xlabel('hours into the box run')
        ax.set_xticks([6, 12, 18, 24])
        if i == 0 and j == 0:
            ax.legend(frameon=False)
fig.suptitle(f'{LAB} vs TOMAS numerical: per-cell total number / mass error (boxes 25-75 %, whiskers 5-95 %)', fontsize=9)
fig.tight_layout(); fig.savefig(f'{outd}/fig3_totals_vs_time{tag}.png', dpi=150); plt.close(fig)

# ---------------- conservation / sanity ----------------
for eng in ('fast', 'emulator'):
    S = lambda t: b[f'{eng}_M'][t].sum(1) * 32 / 96 + b[f'{eng}_S'][t] * 32 / 64 + b[f'{eng}_H'][t] * 32 / 98
    ds = (S(4) - S(0)) / S(0)
    summ[f'sulfur_cons|{eng}'] = dict(median=float(np.median(ds)), max_abs=float(np.abs(ds).max()),
                                      p99_abs=float(np.percentile(np.abs(ds), 99)))
    summ[f'finite|{eng}'] = bool(all(np.isfinite(b[f'{eng}_{q}']).all() for q in 'NMSH'))
    summ[f'nonneg|{eng}'] = bool(all((b[f'{eng}_{q}'] >= 0).all() for q in 'NMSH'))
    if f'{eng}_time' in b.files: summ[f'time_s|{eng}'] = [float(x) for x in b[f'{eng}_time']]
for pn, sl in pop.items():
    for q in 'NM':
        n0 = b[f'fast_{q}'][0][sl].sum(1); n24 = b[f'fast_{q}'][4][sl].sum(1)
        summ[f'{pn}|numerical_change_tot{q}_24h'] = dict(median=float(np.median(n24 / n0 - 1) * 100),
                                                         p10=float(np.percentile(n24 / n0 - 1, 10) * 100),
                                                         p90=float(np.percentile(n24 / n0 - 1, 90) * 100))
    for eng in ('fast', 'emulator'):
        s0 = b[f'{eng}_S'][0][sl]; s4 = b[f'{eng}_S'][4][sl]
        h4n = b['fast_H'][4][sl]; h4e = b['emulator_H'][4][sl]
    summ[f'{pn}|SO2_24h_relerr_median_abs'] = float(np.median(np.abs(b['emulator_S'][4][sl] / np.maximum(b['fast_S'][4][sl], 1e-40) - 1)) * 100)
    summ[f'{pn}|H2SO4_24h_relerr_median_abs'] = float(np.median(np.abs(h4e / np.maximum(h4n, 1e-40) - 1)) * 100)
    summ[f'{pn}|H2SO4_24h_relerr_p90_abs'] = float(np.percentile(np.abs(h4e / np.maximum(h4n, 1e-40) - 1), 90) * 100)
route = np.concatenate([c['bg_route'], c['sai_route']])
quiet = b['triage_quiet'] if 'triage_quiet' in b.files else np.zeros(N, bool)
for pn, sl in pop.items():
    rN = np.abs(b['emulator_N'][4][sl].sum(1) / b['fast_N'][4][sl].sum(1) - 1) * 100
    rt = route[sl] > 0; q_ = quiet[sl] & ~rt
    summ[f'{pn}|routing'] = dict(frac_phys_static=float(rt.mean()), frac_top=float((route[sl] == 1).mean()),
                                 frac_num=float((route[sl] == 2).mean()), frac_coarse=float((route[sl] == 3).mean()),
                                 frac_triage_quiet=float(q_.mean()), frac_network=float((~rt & ~q_).mean()),
                                 totN24_p90_routed=float(np.percentile(rN[rt], 90)) if rt.any() else None,
                                 totN24_p90_network=float(np.percentile(rN[~rt & ~q_], 90)),
                                 totN24_p90_quiet=float(np.percentile(rN[q_], 90)) if q_.any() else None,
                                 frac_of_cells_err_gt10_that_are_routed=float(rt[rN > 10].mean()) if (rN > 10).any() else None,
                                 n_err_gt10=int((rN > 10).sum()))
if 'route' in b.files:
    R = b['route']                                     # (4, N): 0 net, 1 static physics, 2 burst physics, 3 quiet, 4 online-corr sample (physics)
    names = {0: 'network', 1: 'physics_static', 2: 'physics_burst', 3: 'triage_quiet', 4: 'physics_corr_sample'}
    for pn, sl in pop.items():
        Rs = R[:, sl]
        anyphys = np.isin(Rs, (1, 2, 4)).any(0); allnet = (Rs == 0).all(0); allq = (Rs == 3).all(0)
        cls = np.where(anyphys, 'physics >=1 call', np.where(allnet, 'network all 4 calls', np.where(allq, 'quiet all 4 calls', 'network/quiet mix')))
        e = dict(per_call_frac={names[c]: [float((Rs[k] == c).mean()) for k in range(4)] for c in names})
        for q in 'NM':
            rr = np.abs(b[f'emulator_{q}'][4][sl].sum(1) / b[f'fast_{q}'][4][sl].sum(1) - 1) * 100
            rb = np.abs(binerr(q)[sl]) * 100
            for cn in ('physics >=1 call', 'network all 4 calls', 'quiet all 4 calls', 'network/quiet mix'):
                m = cls == cn
                if m.any():
                    e.setdefault(cn, dict(frac=float(m.mean())))
                    e[cn][f'tot{q}_median_abs'] = float(np.median(rr[m])); e[cn][f'tot{q}_p90_abs'] = float(np.percentile(rr[m], 90))
                    e[cn][f'bin{q}_median_abs'] = float(np.nanmedian(rb[m])) if np.isfinite(rb[m]).any() else None
                    e[cn][f'bin{q}_p90_abs'] = float(np.nanpercentile(rb[m], 90)) if np.isfinite(rb[m]).any() else None
        summ[f'{pn}|actual_routes'] = e
json.dump(summ, open(f'{outd}/summary{tag}.json', 'w'), indent=1)
for k, v in summ.items():
    if 'bin_' in k:
        v = {kk: vv for kk, vv in v.items() if kk.startswith('all_')}
    print(k, v)
