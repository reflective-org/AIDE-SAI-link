#!/usr/bin/env python3
"""The Planck term of the evolved call, hr_clear(T + dT) - hr_clear(T), as an
exact per-column Jacobian: J[k, j] = d hr_k / d T_j on the band levels, by
finite differences (one zero-aerosol solve per band level). Checks, on the
day-30 dT_rad of the physics run: (1) J.dT vs the exact solve at T+dT and
T+2dT (first-order accuracy); (2) the row-sum x local-dT shortcut for the
record; (3) J's band structure, and how many solves a graph-colouring probe
needs to recover it to a given accuracy. GPU + archive, ~3 min.
"""
import os, sys, time
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402
import coupling as C
import radiation as R
import jax, jax.numpy as jnp

st = np.load(f'{_REPO}/runs/bench_phys30d/coupled_state_bench_phys30d_ckpt.npz')
dT = np.asarray(st['dT_rad']); hour = 720
dU = C.open_var('U'); lat = dU['lat'].values; lon = dU['lon'].values
plev = dU['hyam'].values * dU['P0'].values + dU['hybm'].values * C.PS_REF
zd = np.load(f'{_REPO}/runs/chemdump/debug_nucleation/premicro_state_step120.npz')
klevs = [int(np.argmin(np.abs(plev - pp))) for pp in zd['plev']]
drv = R.RadiationDriver(C.open_var, klevs, lat, lon, C.DP_BIN * 1e-9, mmid_kg=C.MMID, rho_dry=C.RHO_AER)
nzb = len(klevs)
T_band = drv._read3d(drv.dT, 'T', hour)[drv.band_pos]
D = 86400.0
hc, dc = drv.heating(hour, None, split=True); hc = np.asarray(hc); hc_sw = np.asarray(dc['hr_sw_band'])
olr_c = np.asarray(dc['olr']); up_c = np.asarray(dc['sw_up_toa'])
t0 = time.time(); EPS = 1.0
J = np.zeros((nzb, nzb, drv.nlat, drv.nlon), np.float32)     # K/day per K
Jsw = np.zeros_like(J)
Jolr = np.zeros((nzb, drv.nlat, drv.nlon), np.float32); Jup = np.zeros_like(Jolr)
for j in range(nzb):
    Tp = T_band.copy(); Tp[j] += EPS
    hp, dp_ = drv.heating(hour, None, T_band_override=Tp, split=True)
    J[:, j] = (np.asarray(hp) - hc) * D / EPS
    Jsw[:, j] = (np.asarray(dp_['hr_sw_band']) - hc_sw) * D / EPS
    Jolr[j] = (np.asarray(dp_['olr']) - olr_c) / EPS
    Jup[j] = (np.asarray(dp_['sw_up_toa']) - up_c) / EPS
print(f'Jacobian: {nzb} solves in {time.time()-t0:.0f}s', flush=True)
# uniform probe (what the cache's hr_dT is) = row sum
rowsum = J.sum(1)

for fac in (1.0, 2.0):
    hr_ex, d_ex = drv.heating(hour, None, T_band_override=T_band + fac * dT)
    exact = (np.asarray(hr_ex) - hc) * D
    lin = np.einsum('kjxy,jxy->kxy', J, fac * dT)
    shortcut = rowsum * (fac * dT)
    for name, est in (('J.dT', lin), ('rowsum*dT', shortcut)):
        err = est - exact
        print(f'[x{fac:.0f}] {name:10s}: |exact| max {np.abs(exact).max():.4f} p99 {np.percentile(np.abs(exact),99):.4f} K/d; '
              f'err max {np.abs(err).max():.5f} p99 {np.percentile(np.abs(err),99):.5f} rms {np.sqrt((err**2).mean()):.2e}; '
              f'rel rms {np.sqrt((err**2).mean())/np.sqrt((exact**2).mean()):.2%}', flush=True)
    ex_olr = np.asarray(d_ex['olr']) - olr_c; li_olr = np.einsum('jxy,jxy->xy', Jolr, fac * dT)
    print(f'[x{fac:.0f}] TOA olr: exact max {np.abs(ex_olr).max():.4f} gmean {ex_olr.mean():.5f} W/m2; J.dT err max {np.abs(li_olr-ex_olr).max():.5f} gmean {np.mean(li_olr-ex_olr):.6f}', flush=True)

# structure
absJ = np.abs(J); tot = absJ.sum((0, 1))
print('J structure (K/day per K): diag mean %.4f, max %.4f; |J| mass within |k-j|<=b:' % (np.abs(np.einsum('kkxy->kxy', J)).mean(), np.abs(np.einsum('kkxy->kxy', J)).max()), flush=True)
kk, jj = np.meshgrid(np.arange(nzb), np.arange(nzb), indexing='ij')
for b in (0, 1, 2, 3, 4, 6, 8, 12):
    m = (np.abs(kk - jj) <= b)
    print(f'    b={b:2d}: {(absJ[m].sum() / absJ.sum()):.4f}', flush=True)
print('   row k: diag, max off-diag, farthest |j-k| with |J|>1% of diag (global max over columns)', flush=True)
for k in range(0, nzb, 3):
    dg = np.abs(J[k, k]); off = absJ[k].copy(); off[k] = 0
    far = max((abs(j - k) for j in range(nzb) if j != k and (absJ[k, j] > 0.01 * dg).any()), default=0)
    print(f'    k={k:2d} ({plev[klevs[k]]/100:6.1f} hPa): diag {dg.mean():.4f}, off max {off.max():.4f}, far {far}', flush=True)
print(f'   SW part of J: |Jsw| / |J| = {np.abs(Jsw).sum()/absJ.sum():.4f}', flush=True)
print(f'   Jolr per level (gmean W/m2/K): ' + ' '.join(f'{v:.3f}' for v in Jolr.mean((1, 2))), flush=True)
print(f'   Jup  per level (gmean W/m2/K): ' + ' '.join(f'{v:.4f}' for v in Jup.mean((1, 2))), flush=True)

# colouring probe: perturb levels j = r mod m together, assign the response to the nearest perturbed level
hr_ex, d_ex = drv.heating(hour, None, T_band_override=T_band + dT)
exact = (np.asarray(hr_ex) - hc) * D
for m in (4, 6, 8, 12):
    Jp = np.zeros_like(J)
    for r in range(m):
        resp = J[:, r::m].sum(1)                     # exact FD response to the combined probe (linear)
        js = np.arange(r, nzb, m)
        for k in range(nzb):
            jn = js[np.argmin(np.abs(js - k))]      # nearest perturbed level
            Jp[k, jn] = resp[k]
    est = np.einsum('kjxy,jxy->kxy', Jp, dT); err = est - exact
    print(f'probe m={m:2d} ({m} solves): J.dT err max {np.abs(err).max():.5f} p99 {np.percentile(np.abs(err),99):.5f} rms {np.sqrt((err**2).mean()):.2e} rel rms {np.sqrt((err**2).mean())/np.sqrt((exact**2).mean()):.2%}', flush=True)
np.savez(os.environ.get('JAC_OUT', '/tmp/jacobian_h720.npz'), J=J, Jsw=Jsw, Jolr=Jolr, Jup=Jup, dT=dT, exact=exact, plev=plev[klevs])
print('PLANCK_JACOBIAN_DONE', flush=True)
