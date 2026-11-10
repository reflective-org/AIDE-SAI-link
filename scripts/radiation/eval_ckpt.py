#!/usr/bin/env python3
"""Offline evaluation of exported radiation-emulator checkpoints on label
shards: per-output-group rms error / signal AND the mean bias, overall and by
column AOD (the global TOA forcing is a mean over the plume, so bias there is
what the in-loop forcing gate feels). Uses the JAX forward of rad_emulator.

    CUDA_VISIBLE_DEVICES=N python3 scripts/radiation/eval_ckpt.py --val DIR --aer A.npz [--planck P.npz]
"""
import os, sys, glob, argparse
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument('--val', nargs='+', required=True); ap.add_argument('--aer', nargs='*', default=[]); ap.add_argument('--planck', nargs='*', default=[])
ap.add_argument('--max-shards', type=int, default=None)
ap.add_argument('--aer-ens', nargs='*', default=[], help='member .npz files evaluated as ONE ensemble (mean output)')
ap.add_argument('--planck-ens', nargs='*', default=[], help='member .npz files evaluated as ONE ensemble (mean output)')
args = ap.parse_args()
import jax, jax.numpy as jnp
import rad_emulator as E
files = sorted(sum((glob.glob(os.path.join(d, 'g*.npz')) for d in args.val), []))[:args.max_shards]
xs, raws, metas = [], [], []
for f in files:
    z = np.load(f); xs.append(z['x']); raws.append(z['raw']); metas.append(z['meta'])
x = np.concatenate(xs); raw = np.concatenate(raws); meta = np.concatenate(metas); nzb = 24
print(f'{len(files)} shards, {len(x)} columns', flush=True)
def targets(raw, net):
    n = nzb; sw1, lw1, swe, lwe, swc, lwc = (raw[:, i*n:(i+1)*n] for i in range(6))
    up1, olr1, upe, olre, upc, olrc = (raw[:, 144 + i] for i in range(6))
    if net == 'aer': return np.concatenate([sw1 - swe, lw1 - lwe, (up1 - upe)[:, None], (olr1 - olre)[:, None]], 1)
    return np.concatenate([swe - swc, lwe - lwc, (upe - upc)[:, None], (olre - olrc)[:, None]], 1)
aod = meta[:, 6]; cosz = meta[:, 7]
def report(path, net):
    ens = isinstance(path, (list, tuple))
    m = E.EnsembleNet(path) if ens else E.ColumnNet(path); y = targets(raw, net)
    if ens: path = os.path.join(os.path.dirname(path[0]), '{' + ','.join(os.path.basename(p) for p in path) + '}')
    pred = np.concatenate([np.asarray(m(jnp.asarray(x[i:i+32768]))) for i in range(0, len(x), 32768)])
    e = pred - y
    print(f'== {net}: {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)} (ep {m.epoch}, head {m.head}, tau_w {m.tau_w})')
    groups = (('SW hr', slice(0, nzb)), ('LW hr', slice(nzb, 2*nzb)), ('swup', slice(2*nzb, 2*nzb+1)), ('olr', slice(2*nzb+1, 2*nzb+2)))
    for lo, hi, nm in ((0, 9, 'all'), (0, 1e-3, 'aod<1e-3'), (1e-3, 1e-2, '1e-3..1e-2'), (1e-2, 5e-2, '1e-2..5e-2'), (5e-2, 9, 'aod>5e-2')):
        sel = (aod >= lo) & (aod < hi)
        if sel.sum() == 0: continue
        parts = []
        for gname, sl in groups:
            ee = e[sel][:, sl]; ss = y[sel][:, sl]
            parts.append(f'{gname} {np.sqrt((ee**2).mean()):.2e}/{np.sqrt((ss**2).mean()):.2e} ({np.sqrt((ee**2).mean())/max(np.sqrt((ss**2).mean()),1e-12):.1%}, bias {ee.mean():+.1e} = {ee.mean()/max(abs(ss.mean()),1e-12):+.1%} of mean {ss.mean():+.1e})')
        print(f'  {nm:11s} n={sel.sum():7d}: ' + ' | '.join(parts), flush=True)
    # TOA: day columns only, bias vs mean
    d = cosz > 0
    print(f'  day columns swup: bias {e[d, 2*nzb].mean():+.2e} W/m2 (mean signal {y[d, 2*nzb].mean():+.3e}); olr bias {e[:, 2*nzb+1].mean():+.2e} (mean {y[:, 2*nzb+1].mean():+.3e})', flush=True)
for p in args.aer: report(p, 'aer')
for p in args.planck: report(p, 'planck')
if args.aer_ens: report(list(args.aer_ens), 'aer')
if args.planck_ens: report(list(args.planck_ens), 'planck')
