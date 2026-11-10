#!/usr/bin/env python3
"""Compare the global trajectories of several runs against a reference run.

    python scripts/validation/compare_runs.py REF_TAG TAG [TAG ...] [--labels L0 L1 ...] [--out NAME]

Reads runs/<TAG>/coupled_timeseries_<TAG>.npz (and micro_bench_<TAG>.npz when
present, for the in-loop engine wall time), prints a day-by-day table of
number and mass burden, SO2 burden, r_eff, AOD and TOA forcing of every TAG
relative to REF_TAG, and writes outputs/emulator_bench/compare_<NAME>.png.
This is the self-rollout benchmark: engine error accumulated through
advection, settling and radiation over the run, as opposed to the one-step
error micro_bench.py measures.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument('tags', nargs='+', help='REF_TAG then the TAGs to compare')
ap.add_argument('--labels', nargs='*', default=None)
ap.add_argument('--out', default=None, help='figure name (default: joined tags)')
ap.add_argument('--days', nargs='*', type=float, default=[1, 5, 10, 20, 30], help='days at which to tabulate')
args = ap.parse_args()
tags = args.tags; labels = args.labels or tags
if len(labels) != len(tags):
    sys.exit('--labels must match the tags')

VARS = [('Nburden', 'number burden', True), ('Mburden', 'mass burden', True), ('SO2burden', 'SO2 burden', True),
        ('reff_nm', 'r_eff [nm]', False), ('aod550', 'AOD 550 nm', False), ('arf_toa_avg', 'TOA forcing [W m-2]', False),
        ('H2SO4burden', 'H2SO4(g) burden', True)]


def load(tag):
    z = np.load(f'{ROOT}/runs/{tag}/coupled_timeseries_{tag}.npz', allow_pickle=True)
    d = {k: np.asarray(z[k]) for k in z.keys()}
    bp = f'{ROOT}/runs/{tag}/micro_bench_{tag}.npz'
    if os.path.exists(bp):
        b = np.load(bp, allow_pickle=True)
        d['t_engine'] = float(np.median(b['t_primary'][2:])); d['t_ref'] = float(np.median(b['t_reference'][2:]))
    return d


runs = [load(t) for t in tags]
ref = runs[0]
n = min(len(r['hours']) for r in runs)
hours = ref['hours'][:n]
day = hours / 24.0
idx = [int(np.argmin(np.abs(day - dd))) for dd in args.days if dd <= day[-1] + 1e-9]

print(f"reference: {tags[0]} ({labels[0]}); {n} steps, {day[-1]:.1f} days")
for r, t, l in zip(runs, tags, labels):
    if 't_engine' in r:
        print(f"  {l:24s} engine {r['t_engine']:6.2f} s/step in loop (its reference engine {r['t_ref']:6.2f} s)")
print()
hdr = 'quantity                  ' + ''.join(f'{"day %g" % day[i]:>12s}' for i in idx)
for key, name, _ in VARS:
    if key not in ref:
        continue
    print(f"{name} (ref value):  " + '  '.join(f'{ref[key][i]:.4g}' for i in idx))
    print(hdr)
    for r, l in list(zip(runs, labels))[1:]:
        if key not in r:
            continue
        rel = (r[key][:n] - ref[key][:n]) / np.where(np.abs(ref[key][:n]) > 0, np.abs(ref[key][:n]), np.nan) * 100
        print(f"  {l:24s}" + ''.join(f'{rel[i]:+11.1f}%' for i in idx))
    print()

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)
fig, axs = plt.subplots(2, 4, figsize=(17, 7.5))
axs = axs.ravel()
for ax, (key, name, norm) in zip(axs, VARS):
    for r, l in zip(runs, labels):
        if key not in r:
            continue
        y = r[key][:n]
        if norm:
            y = y / r[key][0] if r[key][0] > 0 else y
        ax.plot(day, y, label=l, lw=1.4)
    ax.set_title(name + (' / initial' if norm else '')); ax.set_xlabel('day'); ax.grid(alpha=.3)
axs[0].legend(fontsize=8)
ax = axs[len(VARS)]
for r, l in list(zip(runs, labels))[1:]:
    rel = (r['Nburden'][:n] - ref['Nburden'][:n]) / ref['Nburden'][:n] * 100
    ax.plot(day, rel, label=f'{l} N', lw=1.4)
    rel = (r['Mburden'][:n] - ref['Mburden'][:n]) / ref['Mburden'][:n] * 100
    ax.plot(day, rel, label=f'{l} M', lw=1.4, ls='--')
ax.axhline(0, color='k', lw=.6); ax.set_title(f'burden error vs {labels[0]} [%]'); ax.set_xlabel('day'); ax.grid(alpha=.3)
ax.legend(fontsize=7)
fig.suptitle('self-rollout trajectories: ' + ', '.join(labels))
fig.tight_layout()
out = args.out or '_'.join(tags[1:])
os.makedirs(f'{ROOT}/outputs/emulator_bench', exist_ok=True)
fig.savefig(f'{ROOT}/outputs/emulator_bench/compare_{out}.png', dpi=130)
print(f"wrote outputs/emulator_bench/compare_{out}.png")
