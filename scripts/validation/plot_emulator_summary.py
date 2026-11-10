#!/usr/bin/env python3
"""One figure page: the production emulator engine against the fast physics over
the 30-day benchmark (same CESM winds, same everything but the microphysics).

Reads runs/<TAG>/coupled_{timeseries,frames}_<TAG>.npz, micro_bench_<TAG>.npz
and the day-30 state dump; writes outputs/emulator_bench/emulator_summary.png (+ npz of the
derived series; OUT=dir overrides).
"""
import os, re, sys, statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = f'{ROOT}/runs'
OUT = os.environ.get('OUT', f'{ROOT}/outputs/emulator_bench')
PHYS, EMU, NET = 'bench_phys30d', 'bench_chem_hyb1num4_30d', 'bench_emu30d'
RHO = 1770.0
C_PHYS, C_EMU, C_NET = 'k', 'tab:red', 'tab:blue'


def ts(tag):
    z = np.load(f'{R}/{tag}/coupled_timeseries_{tag}.npz', allow_pickle=True)
    return {k: np.asarray(z[k]) for k in z.files}


def frames(tag):
    z = np.load(f'{R}/{tag}/coupled_frames_{tag}.npz', allow_pickle=True)
    return {k: np.asarray(z[k]) for k in z.files}


def mb(tag):
    p = f'{R}/{tag}/micro_bench_{tag}.npz'
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {k: np.asarray(z[k]) for k in z.files}


def rel(a, b):
    return 100.0 * (a - b) / np.abs(b)


T = {k: ts(k) for k in (PHYS, EMU, NET)}
F = {k: frames(k) for k in (PHYS, EMU, NET)}
B = {k: mb(k) for k in (PHYS, EMU)}
lat = F[PHYS]['lat']; nlon = F[PHYS]['frames_mas'].shape[-1]; lon = np.arange(nlon) * 360.0 / nlon
xk = F[PHYS]['xk']; dp_edge = np.cbrt(6.0 * xk / (np.pi * RHO)) * 1e9          # nm
dp_mid = np.sqrt(dp_edge[:-1] * dp_edge[1:]); dlog = np.log10(dp_edge[1:] / dp_edge[:-1])
probe = float(F[PHYS]['probe_hpa'])
days_f = F[PHYS]['frame_hours'] / 24.0
area = np.cos(np.radians(lat))

fig, axs = plt.subplots(5, 3, figsize=(19, 25))
fig.subplots_adjust(hspace=0.42, wspace=0.28, top=0.95, bottom=0.03, left=0.05, right=0.98)

# ---- row 1: global trajectories ---------------------------------------------
for ax, (key, lab, scale) in zip(axs[0, :2], [('Nburden', 'number burden / N(t=0)', None), ('aod550', 'global AOD 550 nm', None)]):
    for tag, c, l in ((PHYS, C_PHYS, 'fast physics'), (EMU, C_EMU, 'emulator, production config'), (NET, C_NET, 'network alone (first pass)')):
        d = T[tag]; y = d[key] / (d[key][0] if key == 'Nburden' else 1.0)
        ax.plot(d['hours'] / 24.0, y, c, lw=1.6 if tag != NET else 1.0, ls='-' if tag != NET else '--', label=l)
    ax.set_xlabel('day'); ax.set_ylabel(lab); ax.grid(alpha=.3)
axs[0, 0].legend(fontsize=8, loc='upper left'); axs[0, 0].set_title('(a) global number burden, 10 Tg/yr SO2 at 55 hPa, ±30°')
axs[0, 1].set_title('(b) global AOD')
ax = axs[0, 2]
dP, dE = T[PHYS], T[EMU]
for key, lab in (('Nburden', 'number'), ('Mburden', 'mass'), ('SO2burden', 'SO2'), ('reff_nm', 'r_eff'), ('aod550', 'AOD'), ('arf_toa_avg', 'TOA forcing')):
    n = min(len(dP[key]), len(dE[key]))
    y = rel(dE[key][:n], dP[key][:n])
    if key == 'arf_toa_avg':
        y = np.where(dE['hours'][:n] / 24.0 < 3, np.nan, y)
    ax.plot(dE['hours'][:n] / 24.0, y, label=lab)
ax.axhline(0, c='k', lw=.6); ax.set_ylim(-12, 12); ax.set_xlabel('day'); ax.set_ylabel('(emulator − physics) / physics  [%]')
ax.set_title('(c) production emulator vs physics, global quantities'); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3)

# ---- row 2: Hovmöller at the probe level ------------------------------------
zm = {k: (F[k]['frames_mas'].sum(1)).mean(-1) for k in (PHYS, EMU)}     # (nframe, nlat) total SO4 mixing ratio
vmax = zm[PHYS].max()
for ax, k, ttl in ((axs[1, 0], PHYS, '(d) physics'), (axs[1, 1], EMU, '(e) emulator')):
    im = ax.pcolormesh(days_f, lat, zm[k].T * 1e9, cmap='viridis', vmin=0, vmax=vmax * 1e9, shading='auto')
    ax.set_xlabel('day'); ax.set_ylabel('latitude'); ax.set_title(f'{ttl}: zonal-mean SO4 at {probe:.0f} hPa [µg/kg]')
    plt.colorbar(im, ax=ax, pad=.02)
ax = axs[1, 2]
dif = 100.0 * (zm[EMU] - zm[PHYS]) / vmax
im = ax.pcolormesh(days_f, lat, dif.T, cmap='RdBu_r', vmin=-2, vmax=2, shading='auto')
ax.set_xlabel('day'); ax.set_ylabel('latitude'); ax.set_title('(f) emulator − physics, % of the physics maximum')
plt.colorbar(im, ax=ax, pad=.02)

# ---- row 3: day-30 maps -----------------------------------------------------
i30 = int(np.argmin(np.abs(days_f - 30)))
col = {k: F[k]['frames_mas'][i30].sum(0) * 1e9 for k in (PHYS, EMU)}
vmax = np.percentile(col[PHYS], 99.5)
for ax, k, ttl in ((axs[2, 0], PHYS, '(g) physics'), (axs[2, 1], EMU, '(h) emulator')):
    im = ax.pcolormesh(lon, lat, col[k], cmap='magma', vmin=0, vmax=vmax, shading='auto')
    ax.set_title(f'{ttl}: SO4 at {probe:.0f} hPa, day {days_f[i30]:.0f} [µg/kg]'); ax.set_xlabel('longitude'); ax.set_ylabel('latitude')
    plt.colorbar(im, ax=ax, pad=.02)
ax = axs[2, 2]
im = ax.pcolormesh(lon, lat, 100.0 * (col[EMU] - col[PHYS]) / vmax, cmap='RdBu_r', vmin=-3, vmax=3, shading='auto')
ax.set_title('(i) emulator − physics, % of the physics 99.5th pct'); ax.set_xlabel('longitude'); ax.set_ylabel('latitude')
plt.colorbar(im, ax=ax, pad=.02)

# ---- row 4: size distributions, one-step error, routing map -------------------
ax = axs[3, 0]; axm = ax.twinx()
bands = {'ring 20–40°': (20, 40), 'diffuse 45–70°': (45, 70)}
for (bl, (a, b)), ls in zip(bands.items(), ('-', '--')):
    m = (np.abs(lat) >= a) & (np.abs(lat) <= b)
    w = area[m] / area[m].sum()
    for k, c, l in ((PHYS, C_PHYS, 'physics'), (EMU, C_EMU, 'emulator'), (NET, C_NET, 'network alone')):
        N = (F[k]['frames_num'][i30][:, m, :].mean(-1) * w[None, :]).sum(1)       # (40,) #/kg
        M = (F[k]['frames_mas'][i30][:, m, :].mean(-1) * w[None, :]).sum(1) * 1e9  # µg/kg
        ax.plot(dp_mid, N / dlog, c, ls=ls, lw=1.4 if k != NET else .9, label=f'{l}, {bl}')
        axm.plot(dp_mid, M / dlog, c, ls=ls, lw=1.4 if k != NET else .9, alpha=.45)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('D_p [nm]'); ax.set_ylabel('dN / dlog10 D_p  [#/kg air]  (bold)')
axm.set_yscale('log'); axm.set_ylabel('dM / dlog10 D_p  [µg/kg]  (faint)')
ax.set_ylim(1e4, None); axm.set_ylim(1e-4, None)
ax.set_title(f'(j) size distribution at {probe:.0f} hPa, day 30'); ax.legend(fontsize=7, loc='lower left'); ax.grid(alpha=.3, which='both')

ax = axs[3, 1]
b = B[EMU]
day_b = (b['step'] + 1) * 6.0 / 24.0
scale = 100.0 if np.nanmax(b['totN_active']) < 5 else 1.0
for key, c, l in (('totN_active', 'tab:red', 'number'), ('totM_active', 'tab:orange', 'mass')):
    ax.plot(day_b, scale * b[key][:, 1], c, label=f'{l}, per-cell p90'); ax.plot(day_b, scale * b[key][:, 0], c, ls=':', label=f'{l}, per-cell p50')
gN = b['totN_global']; gN = gN * (100.0 if np.nanmax(np.abs(gN)) < 5 else 1.0)
ax.plot(day_b, np.abs(gN), 'k', lw=1.6, label='|global number|')
ax.set_yscale('log'); ax.set_xlabel('day'); ax.set_ylabel('one-step |emulator − physics| / physics  [%]')
ax.set_title('(k) one-step error on the same input, active cells'); ax.legend(fontsize=7); ax.grid(alpha=.3, which='both')

# routing map from the day-30 state dump (the adapter's own ranking)
ax = axs[3, 2]
dump = f'{R}/chemdump/debug_nucleation/premicro_state_step120.npz'
if os.path.exists(dump):
    z = np.load(dump)
    T3 = z['T3d']; P3 = z['pres3d']; rho = P3 / (287.05 * T3)
    num = z['num']; mas = z['mas']; nlev, nlat_, nlon_ = T3.shape
    Nk = num * rho[None]; Nk = np.where(Nk > 1e-10, Nk, 0.0)
    Mt = np.minimum(np.maximum(mas * rho[None], Nk * xk[:-1, None, None, None]), Nk * xk[1:, None, None, None])
    sulf = Mt.sum(0) + z['so2'] * rho * (96.06 / 64.06) + z['h2so4'] * rho * (96.06 / 98.08)
    ncell = sulf.size; s = sulf.ravel(); n = Nk.sum(0).ravel()
    k1 = max(1, int(round(0.01 * ncell))); k2 = max(1, int(round(0.04 * ncell)))
    sel1 = np.argpartition(-s, k1)[:k1]; n2 = n.copy(); n2[sel1] = -1; sel2 = np.argpartition(-n2, k2)[:k2]
    route = np.zeros(ncell); route[sel1] = 1; route[sel2] = 2
    route = route.reshape(nlev, nlat_, nlon_)
    frac = (route > 0).mean(-1) * 100.0                                                 # (nlev, nlat)
    plev = z['plev'] / 100.0
    im = ax.pcolormesh(lat, plev, frac, cmap='Purples', vmin=0, vmax=max(1.0, frac.max()), shading='auto')
    ax.set_yscale('log'); ax.invert_yaxis(); ax.set_xlabel('latitude'); ax.set_ylabel('pressure [hPa]')
    ax.set_title(f'(l) cells handed to the physics, day 30 [% of longitudes]\n1 % by sulfur (plume) + 4 % by number (dense fine modes)')
    plt.colorbar(im, ax=ax, pad=.02)
    np.savez(f'{OUT}/routing_map_day30.npz', frac=frac, lat=lat, plev=plev,
             frac_plume=(route == 1).mean(-1) * 100, frac_num=(route == 2).mean(-1) * 100)
    del num, mas, Nk, Mt
else:
    ax.text(.5, .5, 'no state dump', ha='center'); ax.set_axis_off()

# ---- row 5: timing, Pareto, r_eff ------------------------------------------
def stage_medians(tag):
    L = open(f'{R}/{tag}/{tag}.log').read().splitlines(); keys = {}
    for l in L:
        m = re.search(r'\[prof\] s=(\d+) (.*)', l)
        if not m or int(m.group(1)) < 2:
            continue
        for k, v in re.findall(r'(\w[\w+]*)=([\d.]+)s', m.group(2)):
            keys.setdefault(k, []).append(float(v))
    return {k: st.median(v) for k, v in keys.items()}


def engine_median(tag):
    b = B.get(tag) or mb(tag)
    if b is None:
        return np.nan
    tp = b['t_primary'][2:]
    return float(np.median(tp))


stg = stage_medians(EMU)                      # all non-micro stages identical between the runs
micro = {'fast physics': engine_median(PHYS) if B[PHYS] is not None else 14.6, 'emulator (production)': engine_median(EMU)}
# in the A/B runs t_primary is the engine that advanced the run: fast in bench_phys30d, emulator in the other
ax = axs[4, 0]
order = [('advect', 'advection'), ('micro', 'microphysics'), ('radiation', 'radiation'), ('bc+polar', 'BC + polar'), ('read', 'CESM read'), ('settle', 'settling')]
cols = ['tab:green', 'tab:red', 'tab:orange', 'tab:gray', 'tab:brown', 'tab:olive']
tot = {}
for i, (name, mval) in enumerate(micro.items()):
    bottom = 0.0
    for (k, l), c in zip(order, cols):
        v = mval if k == 'micro' else stg.get(k, 0.0)
        ax.bar(i, v, bottom=bottom, color=c, label=l if i == 0 else None, width=.6)
        if v > 1.5:
            ax.text(i, bottom + v / 2, f'{v:.1f}', ha='center', va='center', color='w', fontsize=9)
        bottom += v
    tot[name] = bottom
    ax.text(i, bottom + .4, f'{bottom:.1f} s', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(micro))); ax.set_xticklabels(list(micro.keys())); ax.set_ylabel('seconds per 6-h step (median over 30 d)')
ax.set_title(f'(m) cost of one coupled step on one H100\nmicrophysics {micro["fast physics"]/micro["emulator (production)"]:.1f}×, whole step {tot["fast physics"]/tot["emulator (production)"]:.2f}×')
ax.legend(fontsize=8, loc='upper right'); ax.set_ylim(0, max(tot.values()) * 1.15)

ax = axs[4, 1]
cfgs = [(NET, 'network alone'), ('bench_hyb1_30d', 'plume 1 %'), ('bench_hyb3_30d', 'plume 3 %'),
        ('bench_chem_hyb1_30d', 'plume 1 % + chem'), ('bench_chem_hyb3_30d', 'plume 3 % + chem'),
        ('bench_chem_hyb1nuc4_30d', '+ H2SO4-rank 4 %'), ('bench_chem_hyb1nuc10_30d', '+ H2SO4-rank 10 %'),
        (EMU, '+ number-rank 4 %\n(production)'), ('bench_chem_hyb1num8_30d', '+ number-rank 8 %'),
        ('bench_ftchem_hyb1_30d', 'fine-tuned ep2'), ('bench_ft2chem_hyb1_30d', 'fine-tuned best')]
pts = []
for tag, l in cfgs:
    try:
        d = ts(tag)
    except FileNotFoundError:
        continue
    n = min(len(d['Nburden']), len(dP['Nburden']))
    if n < 100:
        continue
    eN = rel(d['Nburden'][n - 1], dP['Nburden'][n - 1]); eA = rel(d['aod550'][n - 1], dP['aod550'][n - 1])
    t = engine_median(tag)
    pts.append((tag, l, t, eN, eA))
for tag, l, t, eN, eA in pts:
    c = C_EMU if tag == EMU else ('tab:purple' if 'ft' in tag else 'tab:blue')
    ax.scatter(t, max(abs(eN), abs(eA)), s=70 if tag == EMU else 35, c=c, zorder=3)
    ax.annotate(l, (t, max(abs(eN), abs(eA))), fontsize=7, xytext=(4, 3), textcoords='offset points')
ax.axvline(micro['fast physics'], c='k', ls='--', lw=1); ax.text(micro['fast physics'], ax.get_ylim()[1] * .9 if ax.get_ylim()[1] > 1 else 30, ' physics', fontsize=8)
ax.set_yscale('log'); ax.set_xlabel('microphysics stage [s / step]'); ax.set_ylabel('max(|number|, |AOD|) error at day 30 [%]')
ax.set_title('(n) accuracy vs cost of the configurations tried'); ax.grid(alpha=.3, which='both')
np.savez(f'{OUT}/pareto_points.npz', pts=np.array([(t, eN, eA) for _, _, t, eN, eA in pts]), labels=np.array([l for _, l, *_ in pts]))

ax = axs[4, 2]
for tag, c, l in ((PHYS, C_PHYS, 'fast physics'), (EMU, C_EMU, 'emulator, production config'), (NET, C_NET, 'network alone')):
    d = T[tag]; ax.plot(d['hours'] / 24.0, d['reff_nm'], c, ls='-' if tag != NET else '--', lw=1.6 if tag != NET else 1.0, label=l)
ax2 = ax.twinx()
for tag, c, l in ((PHYS, C_PHYS, None), (EMU, C_EMU, None), (NET, C_NET, None)):
    d = T[tag]; ax2.plot(d['hours'] / 24.0, d['arf_toa_avg'], c, ls=':', lw=1.2)
ax2.set_ylabel('TOA forcing [W m⁻²]  (dotted)')
ax.set_xlabel('day'); ax.set_ylabel('global r_eff [nm]  (solid)'); ax.set_title('(o) effective radius and TOA forcing'); ax.legend(fontsize=8); ax.grid(alpha=.3)

fig.suptitle('TOMAS emulator in AIDE-SAI-link: production configuration vs the fast physics, 30-day SAI scenario under CESM meteorology', fontsize=14)
fig.savefig(f'{OUT}/emulator_summary.png', dpi=110)
print('day-30 table:')
for tag, l, t, eN, eA in pts:
    print(f'  {l:28s} engine {t:5.2f} s   N {eN:+6.1f} %   AOD {eA:+6.1f} %')
print('stages', stg, 'micro', micro, 'tot', tot)
print('FIG_DONE')
