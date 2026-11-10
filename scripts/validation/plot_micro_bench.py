#!/usr/bin/env python3
"""Figures for a microphysics-engine A/B (driver_fast.py MICRO_REF=...).

Reads, from the current working directory (the run directory, like plot_run.py):
    micro_bench_<TAG>.npz          the per-step one-step comparison and timings
    coupled_timeseries_<TAG>.npz   the run's own scalar history
and optionally a SECOND tag whose time series is overlaid, so the two
trajectories (physics-driven vs emulator-driven) can be compared:

    python3 plot_micro_bench.py emu10d --vs phys10d

Writes <TAG>_bench_onestep.png, <TAG>_bench_bins.png, <TAG>_bench_timing.png,
<TAG>_bench_zonal.png and, with --vs, <TAG>_bench_traj.png.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('tag')
ap.add_argument('--vs', default=None, help='second OUT_TAG to overlay in the trajectory figure')
ap.add_argument('--label', default=None, help='label for TAG (default: its primary engine)')
ap.add_argument('--vs-label', default=None)
args = ap.parse_args()
TAG = args.tag

b = np.load(f'micro_bench_{TAG}.npz')
prim = str(b['primary']); ref = str(b['reference'])
# the error is always |other - truth| / truth, whichever role the truth plays
truth = str(b['truth']) if 'truth' in b.files else ref
other = ref if truth == prim else prim
steps = b['step']; hours = (steps + 1) * 6.0; days = hours / 24.0
lab = args.label or f'{prim} (advances)'

# ---- 1. one-step error over time --------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for a, (key, ttl) in zip(ax.ravel(), (('totN', 'total number'), ('totM', 'total SO4 mass'),
                                      ('so2', 'SO2 gas'), ('h2so4', 'H2SO4 gas'))):
    for sub, ls in (('active', '-'), ('plume', '--')):
        p = b[f'{key}_{sub}'] * 100
        a.plot(days, p[:, 0], ls, color='C0', label=f'{sub} p50')
        a.plot(days, p[:, 1], ls, color='C1', label=f'{sub} p90')
    a.plot(days, np.abs(b[f'{key}_global']) * 100, ':', color='k', label='|global burden-wtd|')
    a.set_yscale('log'); a.set_title(f'{ttl}: one-step |{other} - {truth}| / {truth}')
    a.set_ylabel('rel. error [%]'); a.grid(alpha=.3)
ax[0, 0].legend(fontsize=7, ncol=2)
for a in ax[1]:
    a.set_xlabel('day')
fig.suptitle(f'{TAG}: per-step microphysics A/B, run advanced by {prim}, reference {ref}, truth {truth} '
             f'(active = truth N > 1 cm$^{{-3}}$; plume = top 1 % cells by gas+aerosol)')
fig.tight_layout(); fig.savefig(f'{TAG}_bench_onestep.png', dpi=130); plt.close(fig)

# ---- 2. per-bin error (time-median of the per-step medians) ----------------
D_UM = 1.7e-3 * 2.0 ** ((np.arange(40) + 0.5) / 3.0)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)          # bins never occupied -> NaN, fine
    p50 = np.nanmedian(b['bin_p50'], 0) * 100; p90 = np.nanmedian(b['bin_p90'], 0) * 100
ax[0].plot(D_UM * 1e3, p50, 'o-', label='p50 (occupied bins)'); ax[0].plot(D_UM * 1e3, p90, 's-', label='p90')
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('bin diameter [nm]')
ax[0].set_ylabel('per-bin number rel. error [%]'); ax[0].set_title('per-bin one-step error (median over steps)')
ax[0].grid(alpha=.3); ax[0].legend()
w1 = b['W1_active']
ax[1].plot(days, w1[:, 0], label='W1 p50'); ax[1].plot(days, w1[:, 1], label='W1 p90')
ax[1].set_xlabel('day'); ax[1].set_ylabel('W1 along bin axis [bins]'); ax[1].grid(alpha=.3); ax[1].legend()
ax[1].set_title('size-distribution shape error (number-fraction CDF)')
fig.tight_layout(); fig.savefig(f'{TAG}_bench_bins.png', dpi=130); plt.close(fig)

# ---- 3. timing --------------------------------------------------------------
tp = b['t_primary']; tr = b['t_reference']
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(days, tp, 'o-', label=f'{prim} (advances the run)'); ax.plot(days, tr, 's-', label=f'{ref} (reference)')
ax.set_yscale('log'); ax.set_xlabel('day'); ax.set_ylabel('microphysics wall time per step [s]')
ss = slice(1, None) if len(tp) > 1 else slice(None)
ax.set_title(f'steady state: {prim} {np.median(tp[ss]):.2f} s, {ref} {np.median(tr[ss]):.2f} s '
             f'-> {np.median(tr[ss]) / max(np.median(tp[ss]), 1e-9):.1f}x  (step 0 carries warm-up)')
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(f'{TAG}_bench_timing.png', dpi=130); plt.close(fig)

# ---- 4. zonal disagreement at the last compared step -------------------------
ts = np.load(f'coupled_timeseries_{TAG}.npz')
nlev = b['zm_N_ref'].shape[1]
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
last = -1
for row, (q, unit) in enumerate((('N', 'cm$^{-3}$'), ('M', 'kg m$^{-3}$'))):
    zr = b[f'zm_{q}_ref'][last]; zp = b[f'zm_{q}_pred'][last]
    vmax = np.nanmax(zr)
    for col, (z, ttl) in enumerate(((zr, f'{truth} (truth)'), (zp, f'{other}'))):
        im = ax[row, col].imshow(z, aspect='auto', origin='upper', vmin=0, vmax=vmax, cmap='viridis')
        ax[row, col].set_title(f'zonal-mean total {q}, {ttl}'); plt.colorbar(im, ax=ax[row, col], label=unit)
    d = (zp - zr) / np.maximum(zr, 1e-30) * 100
    im = ax[row, 2].imshow(d, aspect='auto', origin='upper', vmin=-50, vmax=50, cmap='RdBu_r')
    ax[row, 2].set_title(f'({other} - {truth}) / {truth}  [%]'); plt.colorbar(im, ax=ax[row, 2])
for a in ax.ravel():
    a.set_xlabel('lat index'); a.set_ylabel('level index (top=0)')
fig.suptitle(f'{TAG}: one-step zonal disagreement at step {int(steps[last])} (day {days[last]:.1f})')
fig.tight_layout(); fig.savefig(f'{TAG}_bench_zonal.png', dpi=130); plt.close(fig)

# ---- 5. trajectories: this run vs another tag ----------------------------------
if args.vs:
    t2 = np.load(f'coupled_timeseries_{args.vs}.npz')
    lab2 = args.vs_label or args.vs
    keys = [('Mburden', 'SO4 burden / M0'), ('Nburden', 'number burden / N0'),
            ('SO2burden', 'SO2 burden'), ('H2SO4burden', 'H2SO4 burden'),
            ('reff_nm', 'r_eff at probe level [nm]'), ('meanDp_mass_nm', 'mass-mean Dp [nm]'),
            ('aod550', 'AOD 550 nm'), ('arf_toa_avg', 'TOA forcing, trailing avg [W m-2]'),
            ('B_micro', 'budget: cumulative micro / M0'), ('settleM_cum', 'cumulative settled mass'),
            ('clipMadd_cum', 'cumulative two-moment clip mass added'),
            ('Nfloor_cum', 'cumulative number created by the floor')]
    keys = [(k, t) for k, t in keys if k in ts.files and k in t2.files]
    n = len(keys); nc = 3; nr = -(-n // nc)
    fig, ax = plt.subplots(nr, nc, figsize=(15, 3.2 * nr))
    for a, (k, ttl) in zip(ax.ravel(), keys):
        for z, l, ls in ((ts, lab, '-'), (t2, lab2, '--')):
            h = z['hours'] / 24.0; y = z[k]
            if k == 'Mburden':
                y = y / z['M0']
            if k == 'Nburden':
                y = y / z['N0']
            a.plot(h, y, ls, label=l)
        a.set_title(ttl); a.grid(alpha=.3); a.set_xlabel('day')
    ax.ravel()[0].legend()
    for a in ax.ravel()[n:]:
        a.axis('off')
    fig.suptitle(f'trajectory comparison: {TAG} ({prim}) vs {args.vs}')
    fig.tight_layout(); fig.savefig(f'{TAG}_bench_traj.png', dpi=130); plt.close(fig)

print(f"wrote {TAG}_bench_onestep.png {TAG}_bench_bins.png {TAG}_bench_timing.png {TAG}_bench_zonal.png"
      + (f" {TAG}_bench_traj.png" if args.vs else ""))
