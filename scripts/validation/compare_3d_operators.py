#!/usr/bin/env python3
# scripts/validation/compare_3d_operators.py -- 2026-10-14. 3-D truth test with a
# DIVERGENT flow for the production transport entry points: Nair-Lauritzen (2010)
# case 3 (divergent, reversing) in a frame rotating once per period, times a
# vertical profile g(p) with dp-weighted mean zero, so the continuity omega is 0
# at both faces (closed column) and the exact mixing ratio at t = T is the
# initial field. Run on a GPU:  KDEF=2 CUDA_VISIBLE_DEVICES=4 python3 <this>
# Result (f09, 24 uniform levels, 6 h steps, 12 d; L2 / surviving peak):
#   KDEF=2: lr 0.0089/0.924  split K6 0.0067/0.935  strang K6 0.0064/0.935  K24 0.0158
#   KDEF=5: lr 0.0250/0.924  split K6 0.0244/0.935  strang K6 0.0234/0.932  K24 0.0305
# => in a CLOSED column the split is at least as accurate as lr; the 30-day A/B
#    gap is entirely at the OPEN bottom face (see sl_lr._strang_substep).
"""3-D truth test with DIVERGENT flow: Nair-Lauritzen (2010) case 3 (divergent,
reversing, cos(pi t/T) envelope) + zonal background, times a vertical profile
g(p) with dp-weighted mean ZERO, so the continuity omega vanishes at both
faces: a closed column, and the exact mixing ratio at t = T is the initial
field. Both production entry points, f32, 6 h steps, 12 days."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')); import _paths  # noqa
import jax, jax.numpy as jnp
import fct_lr, sl_lr, ffsl_lr
from fct_fast import grid_metric

RAD = 6.371e6; T = 12 * 86400.0; DT = 6 * 3600.0; NSTEP = 48
LAT = np.linspace(-90.0, 90.0, 192); NLON = 288; LON = np.arange(NLON) * 360.0 / NLON
lon = np.broadcast_to(np.deg2rad(LON)[None, :], (192, NLON)); lat = np.broadcast_to(np.deg2rad(LAT)[:, None], (192, NLON))
NLEV = 24; plev = np.linspace(2000.0, 14300.0, NLEV); dp = np.gradient(plev)
g = np.sin(2 * np.pi * (np.arange(NLEV) + 0.5) / NLEV); g = g - (g * dp).sum() / dp.sum()   # sum(dp g) = 0
_, ac = grid_metric(LAT, None); W3 = dp[:, None, None] * ac[None, :, None]

def bell(lon_c, lat_c, R):
    r = RAD * np.arccos(np.clip(np.sin(lat_c) * np.sin(lat) + np.cos(lat_c) * np.cos(lat) * np.cos(lon - lon_c), -1, 1))
    return np.where(r < R, 0.5 * (1 + np.cos(np.pi * r / R)), 0.0)

def wind2d(t, K=float(os.environ.get("KDEF", "2"))):
    # deformational (divergent, reversing) part only; the background rotation is added in wind3d, uniform in z
    k = K * RAD / T; lp = lon - 2 * np.pi * t / T; c = np.cos(np.pi * t / T)
    u = -k * np.sin(lp / 2) ** 2 * np.sin(2 * lat) * np.cos(lat) ** 2 * c
    v = 0.5 * k * np.sin(lp) * np.cos(lat) ** 3 * c
    return u, v

UBG = 2 * np.pi * RAD * np.cos(lat) / T                                 # solid body, one turn in T, non-divergent
def wind3d(t):
    u, v = wind2d(t)
    return (g[:, None, None] * u[None] + UBG[None]).astype(np.float32), (g[:, None, None] * v[None]).astype(np.float32)

hz = 0.9 * (bell(5 * np.pi / 6, 0.0, RAD / 2) + bell(7 * np.pi / 6, 0.0, RAD / 2))
vz = np.exp(-((np.arange(NLEV) - 11.5) / 4.0) ** 2)
Q0 = (0.05 + vz[:, None, None] * hz[None])[None]                     # (1, nlev, nlat, nlon)
Z = np.zeros((NLEV, 192, NLON), np.float32)
M0 = (Q0[0] * W3).sum()

def run(scheme, K=6, hourly=True, rho=False, split='lie'):
    q = Q0.copy(); qfroz = np.zeros_like(q); t0 = time.time(); nsubs = []
    for s in range(NSTEP):
        ta = s * DT; tb = ta + DT
        ua, va = wind3d(ta); ub, vb = wind3d(tb)
        if scheme == 'lr':
            q, n = fct_lr.advect_step_batch(q, ua, va, Z, ub, vb, Z, LAT, dp, qfroz, DT, lat_freeze=80.0, cfl=0.5, dtype=jnp.float32)
        elif scheme == 'ffsl':
            kw = dict(lat_freeze=80.0, nsub=K, dtype=jnp.float32, split=split)
            if hourly:
                kw['hourly_winds'] = [(ua, va, Z)] + [(*wind3d(ta + 3600.0 * k), Z) for k in range(1, 6)] + [(ub, vb, Z)]
            q, n = ffsl_lr.advect_step_batch(q, ua, va, Z, ub, vb, Z, LAT, dp, qfroz, DT, **kw)
        else:
            kw = dict(lat_freeze=80.0, nsub=K, fixer='global', vert='flux', plev=plev, dtype=jnp.float32, rho=rho, split=split)
            if hourly:
                kw['hourly_winds'] = [(ua, va, Z)] + [(*wind3d(ta + 3600.0 * k), Z) for k in range(1, 6)] + [(ub, vb, Z)]
            q, n = sl_lr.advect_step_batch(q, ua, va, Z, ub, vb, Z, LAT, dp, qfroz, DT, **kw)
        q = np.asarray(q, np.float64); nsubs.append(int(n))
    el = (time.time() - t0) / NSTEP
    f = q[0]; h = Q0[0]
    l2 = np.sqrt((W3 * (f - h) ** 2).sum() / (W3 * (h - 0.05) ** 2).sum())
    M1 = (f * W3).sum()
    # vertical structure error: the dp-weighted column profile at the bell centres' band
    prof = (ac[None, :, None] * f).sum(axis=(1, 2)) / (ac[None, :, None] * h).sum(axis=(1, 2))
    print(f"{scheme:>2s} K={K:<2d} hourly={int(hourly)} rho={int(rho)} {split:6s} L2 {l2:.4f}  max {f.max():.4f}  min {f.min():+.3e}  "
          f"dM/M0 {(M1 - M0) / M0:+.1e}  level-mass ratio mid {prof[11]:.3f} edges {prof[0]:.3f}/{prof[-1]:.3f}  {el:.2f} s/step (nsub {np.mean(nsubs):.0f})", flush=True)

print(f"exact: max {Q0.max():.4f}; omega at the faces from continuity = 0 by construction (sum dp g = {np.abs((dp*g).sum()):.1e})")
run('lr')
run('sl', 6, True, False)
run('sl', 6, True, True)
run('sl', 6, True, True, 'strang')
run('sl', 24, True, True, 'strang')
run('sl', 2, True, True, 'strang')
run('ffsl', 6, True, False, 'strang')
run('ffsl', 24, True, False, 'strang')
run('ffsl', 6, True, False, 'lie')
