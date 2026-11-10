#!/usr/bin/env python3
"""Score one campaign run against another from their frames3d/ step files.

Made for run 6 (passive-tracer transport) vs run 2 (full per-bin transport),
but any two runs with FRAME3D=1 and the same grid work: the per-step burdens
(area- and dp-weighted, like coupling.py's Mbur), the global AOD and forcing,
and the per-bin relative error of the 3-D fields, over the steps both runs
have written so far (a running comparison while the runs are still going).

  python3 scripts/validation/compare_frames3d.py --a runs/.../run2 --b runs/.../run6 --out run6_vs_run2 [--every 4]

Writes <out>.txt and <out>.png in the current directory.
"""
import argparse
import glob
import os

import numpy as np


def wlat(lat):
    w = np.cos(np.deg2rad(lat)); return w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='reference run dir (has frames3d/)')
    ap.add_argument('--b', required=True, help='test run dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=int, default=4)
    a = ap.parse_args()
    fa = sorted(glob.glob(os.path.join(a.a, 'frames3d', 'step_*.npz')))
    fb = sorted(glob.glob(os.path.join(a.b, 'frames3d', 'step_*.npz')))
    common = sorted(set(map(os.path.basename, fa)) & set(map(os.path.basename, fb)))[::a.every]
    if not common:
        raise SystemExit('no common steps yet')
    z0 = np.load(os.path.join(a.a, 'frames3d', common[0]))
    lat = z0['lat']; plev = z0['plev_pa']; w = wlat(lat)
    dp = np.gradient(plev)[:, None, None] * w[None, :, None]            # burden weight (nlev,nlat,1)
    rows = []
    bin_err = []
    for name in common:
        A = np.load(os.path.join(a.a, 'frames3d', name)); B = np.load(os.path.join(a.b, 'frames3d', name))
        day = float(A['hour']) / 24.0
        mA = (A['mtot'] * dp).sum(); mB = (B['mtot'] * dp).sum()
        nA = (A['ntot'] * dp).sum(); nB = (B['ntot'] * dp).sum()
        rec = dict(day=day, dM=mB / mA - 1, dN=nB / nA - 1)
        if 'aod550' in A.files:
            rec['aodA'] = float((A['aod550'] * w[:, None]).sum() / A['aod550'].shape[1])
            rec['aodB'] = float((B['aod550'] * w[:, None]).sum() / B['aod550'].shape[1])
            rec['arfA'] = float((A['toa_arf'] * w[:, None]).sum() / A['toa_arf'].shape[1]) if 'toa_arf' in A.files else np.nan
            rec['arfB'] = float((B['toa_arf'] * w[:, None]).sum() / B['toa_arf'].shape[1]) if 'toa_arf' in B.files else np.nan
        # per-bin mass burden error and the 3-D rms error relative to the field's rms
        ma = A['mas'].astype(np.float64); mb = B['mas'].astype(np.float64)
        bm_a = (ma * dp[None]).sum(axis=(1, 2, 3)); bm_b = (mb * dp[None]).sum(axis=(1, 2, 3))
        rel = np.sqrt(((mb - ma) ** 2 * dp[None]).sum(axis=(1, 2, 3)) / np.maximum((ma ** 2 * dp[None]).sum(axis=(1, 2, 3)), 1e-300))
        bin_err.append((day, bm_b / np.maximum(bm_a, 1e-300) - 1, rel))
        d_eff = (A['deff_wet_nm'], B['deff_wet_nm']); has = A['ntot'] > 0
        rec['deff_rel'] = float(np.sqrt((((d_eff[1] - d_eff[0]) ** 2) * has * dp).sum() / np.maximum(((d_eff[0] ** 2) * has * dp).sum(), 1e-300)))
        rows.append(rec)
    lines = [f'reference {a.a}\ntest      {a.b}\nsteps compared {len(common)} (every {a.every})',
             'day     M_b/M_a-1   N_b/N_a-1   AOD_a    AOD_b    ARF_a    ARF_b    deff rel rms']
    for r in rows:
        lines.append(f"{r['day']:6.1f}  {r['dM']:+9.4f}  {r['dN']:+9.4f}  "
                     + (f"{r['aodA']:.5f}  {r['aodB']:.5f}  {r['arfA']:+8.4f} {r['arfB']:+8.4f}  " if 'aodA' in r else ' ' * 38)
                     + f"{r['deff_rel']:.4f}")
    day, bb, rel = bin_err[-1]
    lines.append(f'\nlast common step (day {day:.1f}): per-bin mass burden ratio-1 and 3-D rel-rms error')
    for k in range(len(bb)):
        lines.append(f'  bin {k:2d}  {bb[k]:+9.4f}   {rel[k]:8.4f}')
    txt = '\n'.join(lines); print(txt)
    open(a.out + '.txt', 'w').write(txt)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        d = [r['day'] for r in rows]
        ax[0].plot(d, [r['dM'] for r in rows], label='mass'); ax[0].plot(d, [r['dN'] for r in rows], label='number')
        ax[0].set_xlabel('day'); ax[0].set_ylabel('test/ref - 1'); ax[0].legend(); ax[0].set_title('global burdens')
        if 'aodA' in rows[0]:
            ax[1].plot(d, [r['aodA'] for r in rows], 'k--', label='ref'); ax[1].plot(d, [r['aodB'] for r in rows], 'r-', label='test')
            ax[1].set_xlabel('day'); ax[1].set_title('global AOD550'); ax[1].legend()
        ax[2].semilogy(range(len(rel)), rel); ax[2].set_xlabel('bin'); ax[2].set_title(f'per-bin 3-D rel rms error, day {day:.0f}')
        fig.tight_layout(); fig.savefig(a.out + '.png', dpi=110); print('wrote', a.out + '.png')
    except Exception as e:  # noqa: BLE001
        print('no figure:', e)


if __name__ == '__main__':
    main()
