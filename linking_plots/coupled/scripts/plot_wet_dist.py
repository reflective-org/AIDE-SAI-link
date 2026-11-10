import glob, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from wet_dist import EDGES, DLN, WT0, DWT, NWT
COL = {'emu': '#eb6834', 'num': '#2a78d6', 'mam': '#1baf7a'}
LAB = {'emu': 'TOMAS emulator', 'num': 'TOMAS numerical', 'mam': 'MAM4 (binned modes)'}
files = sorted(glob.glob(f'{CACHE}/wetdist/v2_day_*.npz')); assert len(files) == NDAYS, len(files)
Z = [dict(np.load(f)) for f in files]
days = np.array([int(z['day']) for z in Z])
ctr = np.sqrt(EDGES[1:] * EDGES[:-1])
def mean(key, sel):
    return np.mean([z[key] for z, s in zip(Z, sel) if s], 0)
ALL = np.ones(NDAYS, bool); L30 = days >= NDAYS - 30
# spread each TOMAS bin uniformly in lnD over its own wet (or dry) edge interval, per wt% class
dpb = np.load(f'{RUNS}/r2_tomas_fast/frames3d/step_0000.npz')['dp_bin_nm']
mmid = (dpb * 1e-9) ** 3 * np.pi / 6 * RHO_AER
xk = mmid[0] / np.sqrt(2.0) * 2.0 ** np.arange(41)
wtc = WT0 + DWT * np.arange(NWT)
Dw_e = np.cbrt(wet_d3(xk[:, None], wtc[None])) * 1e9          # (41, NWT) nm, wet bin edges
Dw_m = np.cbrt(wet_d3(mmid[:, None], wtc[None])) * 1e9         # (40, NWT) nm, bin-mass wet diameter
Dd_e = np.cbrt(xk / RHO_AER * 6 / np.pi) * 1e9                 # (41,) dry edges
lE = np.log(EDGES)
def frac(lo, hi):
    f = np.clip((lE[None] - np.log(lo)[..., None]) / (np.log(hi) - np.log(lo))[..., None], 0, 1)
    return np.diff(f, axis=-1)
Fw = frac(Dw_e[:-1].ravel(), Dw_e[1:].ravel()).reshape(40, NWT, -1)   # (40,NWT,80)
Fd = frac(Dd_e[:-1], Dd_e[1:])                                      # (40,80)
WQ = {'N': np.ones_like(Dw_m), 'S': np.pi * (Dw_m * 1e-9) ** 2, 'V': np.pi / 6 * (Dw_m * 1e-9) ** 3 * 1e6}
WQd = {'N': np.ones(40), 'S': np.pi * (dpb * 1e-9) ** 2, 'V': np.pi / 6 * (dpb * 1e-9) ** 3 * 1e6}
_cache = {}
def spec(k, role, q, sel, dry=False):
    key = (k, role, id(sel))
    if key not in _cache:
        _cache[key] = np.mean([z[f'{k}_{role}_Nkc'] for z, s in zip(Z, sel) if s], 0)
    Nkc = _cache[key]
    if dry:
        return ((Nkc.sum(1) * WQd[q])[:, None] * Fd).sum(0)
    return np.einsum('kc,kcj->j', Nkc * WQ[q], Fw)
stats = {}
def scal(k, role, sel):
    s = mean(f'{k}_{role}_sums', sel)
    return dict(deff_wet=s[1] / s[2], dmass_wet=np.exp(s[4] / s[3]), gf_num=s[5] / s[0], gf_mass=s[7] / s[6],
                deff_dry=s[8] / s[9], N=s[0])
for per, sel in (('annual', ALL), ('last30', L30)):
    for k in COL:
        for role in ('inj', 'ctl'):
            stats[f'{per}_{k}_{role}'] = {a: float(b) for a, b in scal(k, role, sel).items()}
    trh = mean('TRH', sel); stats[f'{per}_T_K'] = float(trh[0]); stats[f'{per}_RH'] = float(trh[1]); stats[f'{per}_wt_pct'] = float(trh[2])
# increment D_eff (inj - ctl sums)
for per, sel in (('annual', ALL), ('last30', L30)):
    for k in COL:
        d = mean(f'{k}_inj_sums', sel) - mean(f'{k}_ctl_sums', sel)
        stats[f'{per}_{k}_incr'] = dict(deff_wet=float(d[1] / d[2]), deff_dry=float(d[8] / d[9]), gf_num=float(d[5] / d[0]), gf_mass=float(d[7] / d[6]))
json.dump(stats, open(f'{CACHE}/wetdist_stats.json', 'w'), indent=1)

QS = (('N', 'number, dN/dlnD$_w$', 'm$^{-2}$'), ('S', 'surface area, dS/dlnD$_w$', 'm$^{2}$ m$^{-2}$'), ('V', 'wet volume, dV/dlnD$_w$', 'cm$^{3}$ m$^{-2}$'))
fig, axs = plt.subplots(3, 3, figsize=(15, 11.5), layout='constrained')
rows = (('annual', ALL, 'inj', ('annual mean' if NDAYS == 365 else '2-year mean') + ', injection runs'), ('last30', L30, 'inj', 'last 30 days, injection runs'),
        ('annual', ALL, 'incr', ('annual mean' if NDAYS == 365 else '2-year mean') + ', injection minus control (SAI increment)'))
for r, (per, sel, role, rlab) in enumerate(rows):
    for c, (q, qlab, u) in enumerate(QS):
        ax = axs[r, c]
        for k in ('num', 'emu', 'mam'):
            y = spec(k, 'inj', q, sel) - (spec(k, 'ctl', q, sel) if role == 'incr' else 0)
            ax.plot(ctr, y / DLN, color=COL[k], lw=1.9, label=LAB[k])
            st = stats[f'{per}_{k}_incr' if role == 'incr' else f'{per}_{k}_inj']
            ax.axvline(st['deff_wet'], color=COL[k], lw=0.9, ls=':', ymax=0.12)
        yd = spec('num', 'inj', q, sel, dry=True) - (spec('num', 'ctl', q, sel, dry=True) if role == 'incr' else 0)
        ax.plot(ctr, yd / DLN, color='0.45', lw=0.9, ls='--', label='TOMAS numerical, DRY diameter')
        ax.set_xscale('log'); ax.set_xlim(1, 3e4); ax.grid(alpha=0.25, which='both')
        if q == 'N' and role != 'incr':
            ax.set_yscale('log'); ax.set_ylim(1e8, None)
        elif q == 'N':
            ax.set_yscale('symlog', linthresh=1e10); ax.set_ylim(-1e12, None)
        ax.axhline(0, color='0.6', lw=0.5)
        ax.set_xlabel('wet diameter D$_w$, nm (dashed grey: dry D)'); ax.set_ylabel(f'{qlab.split(",")[1].strip()}, {u}')
        ax.set_title(f'{qlab.split(",")[0]}: {rlab}', fontsize=9)
    txt = '  '.join(f'{ {"emu": "emu", "num": "num", "mam": "MAM4"}[k]} D$_{{eff,w}}$={(stats[f"{per}_{k}_incr" if role == "incr" else f"{per}_{k}_inj"])["deff_wet"]:.0f} nm' for k in ('emu', 'num', 'mam'))
    axs[r, 2].text(0.02, 0.97, txt.replace('  ', '\n'), transform=axs[r, 2].transAxes, va='top', fontsize=8)
axs[0, 0].legend(frameon=False, fontsize=8, loc='lower left')
fig.suptitle('Global-mean STRATOSPHERIC size distribution vs WET diameter (column-integrated over stratospheric band cells, area-weighted). '
             'Dotted ticks: wet D$_{eff}$ per scheme.\nWater uptake: the SAME prescription for all three schemes - Tabazadeh 1997 equilibrium wt% '
             '(emulator band T, CESM RELHUM climatology) + Tang density on each bin\'s mass, as in the TOMAS runs. '
             'MAM4 (modes cut onto the 40 TOMAS bins) is NOT using its own hygroscopic growth here.\nOne frame per day, all ' + str(NDAYS) + ' days. Each TOMAS bin is spread uniformly in lnD over its own wet edge interval (wt% in 0.25 % classes), plotted on 80 log bins 1 nm - 30 um.', fontsize=9)
fig.savefig(f'{OUT}/figs_v2/fig4_f_wet_size_distribution.png', dpi=150, bbox_inches='tight')
print(json.dumps(stats, indent=1))
