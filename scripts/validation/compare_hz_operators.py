#!/usr/bin/env python3
# scripts/validation/compare_hz_operators.py -- 2026-10-14. Which horizontal transport
# operator is nearer the TRUTH: fct_lr (Lin-Rood PPM, CFL sub-steps) vs sl_lr (semi-
# Lagrangian cubic + quasi-monotone limiter) on analytic test cases. Run on a GPU:
#   CUDA_VISIBLE_DEVICES=4 python scripts/validation/compare_hz_operators.py
# Result (f09, 6 h steps, 12 d): SL K=6 L2 0.0015/0.0125/0.058 vs lr 0.019/0.013/0.13;
# SL K=24 is WORSE (0.062/0.036/0.23): each extra SL sub-step adds interpolation
# smoothing, so K must match the wind cadence (hourly -> K=6), never exceed it.
"""Horizontal-operator accuracy against ANALYTIC solutions, same driver entry
points as production (fct_lr.advect_step_batch vs sl_lr.advect_step_batch, w=0,
lat_freeze=80 caps stirred, f32, 6 h coupling steps, 12-day period).
  W1-a0 / W1-a45 : Williamson (1992) case 1 solid body, one revolution
  NL4            : Nair & Lauritzen (2010) case 4 deformational flow with zonal
                   background, reverses at T/2 -> exact answer = initial field
Prints L2 error, max (1 = exact), min, and the mass change."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')); import _paths  # noqa
import jax, jax.numpy as jnp
import fct_lr, sl_lr, ffsl_lr
from fct_fast import grid_metric

RAD = 6.371e6; T = 12 * 86400.0; DT = 6 * 3600.0; NSTEP = 48
LAT = np.linspace(-90.0, 90.0, 192); NLON = 288; LON = np.arange(NLON) * 360.0 / NLON
lon = np.deg2rad(LON)[None, :]; lat = np.deg2rad(LAT)[:, None]          # (nlat,1),(1,nlon)
lon, lat = np.broadcast_to(lon, (192, NLON)), np.broadcast_to(lat, (192, NLON))
NLEV = 3; plev = np.array([40.0, 50.0, 60.0]) * 100; dp = np.gradient(plev)
_, ac = grid_metric(LAT, None)

def bell(lon_c, lat_c, R):
    r = RAD * np.arccos(np.clip(np.sin(lat_c) * np.sin(lat) + np.cos(lat_c) * np.cos(lat) * np.cos(lon - lon_c), -1, 1))
    return np.where(r < R, 0.5 * (1 + np.cos(np.pi * r / R)), 0.0)

def w1_wind(alpha, t):
    a = np.deg2rad(alpha); u0 = 2 * np.pi * RAD / T
    return (u0 * (np.cos(lat) * np.cos(a) + np.sin(lat) * np.cos(lon) * np.sin(a)), -u0 * np.sin(lon) * np.sin(a))

def nl4_wind(_, t):
    k = 10.0 * RAD / T; lp = lon - 2 * np.pi * t / T; c = np.cos(np.pi * t / T)
    return (k * np.sin(lp) ** 2 * np.sin(2 * lat) * c + 2 * np.pi * RAD * np.cos(lat) / T,
            k * np.sin(2 * lp) * np.cos(lat) * c)

CASES = {'W1-a0': (lambda t: w1_wind(0.0, t), bell(1.5 * np.pi, 0.0, RAD / 3)),
         'W1-a45': (lambda t: w1_wind(45.0, t), bell(1.5 * np.pi, 0.0, RAD / 3)),
         'NL4': (lambda t: nl4_wind(0, t), 0.1 + 0.9 * (bell(5 * np.pi / 6, 0.0, RAD / 2) + bell(7 * np.pi / 6, 0.0, RAD / 2)))}

def pack(h):
    q = np.broadcast_to(h, (NLEV, 192, NLON))[None].astype(np.float64)   # (1,nlev,nlat,nlon)
    return np.ascontiguousarray(q)

def run(case, scheme, K=6, hourly=True):
    windf, h0 = CASES[case]; q = pack(h0); z = np.zeros((NLEV, 192, NLON), np.float32)
    qfroz = np.zeros_like(q); M0 = (q[0] * dp[:, None, None] * ac[None, :, None]).sum()
    W = lambda t: tuple(np.broadcast_to(x, (NLEV, 192, NLON)).astype(np.float32) for x in windf(t))
    t0 = time.time()
    for s in range(NSTEP):
        ta = s * DT; tb = ta + DT
        ua, va = W(ta); ub, vb = W(tb)
        if scheme == 'lr':
            q, n = fct_lr.advect_step_batch(q, ua, va, z, ub, vb, z, LAT, dp, qfroz, DT,
                                            lat_freeze=80.0, cfl=0.5, dtype=jnp.float32)
        elif scheme == 'ffsl':
            kw = dict(lat_freeze=80.0, nsub=K, dtype=jnp.float32)
            if hourly:
                kw['hourly_winds'] = [(ua, va, z)] + [(*W(ta + 3600.0 * k), z) for k in range(1, 6)] + [(ub, vb, z)]
            q, n = ffsl_lr.advect_step_batch(q, ua, va, z, ub, vb, z, LAT, dp, qfroz, DT, **kw)
        else:
            kw = dict(lat_freeze=80.0, nsub=K, fixer='global', vert='flux', plev=plev, dtype=jnp.float32)
            if hourly:
                kw['hourly_winds'] = [(ua, va, z)] + [(*W(ta + 3600.0 * k), z) for k in range(1, 6)] + [(ub, vb, z)]
            q, n = sl_lr.advect_step_batch(q, ua, va, z, ub, vb, z, LAT, dp, qfroz, DT, **kw)
        q = np.asarray(q, np.float64)
    el = (time.time() - t0) / NSTEP
    f = q[0, 1]; w = np.cos(lat)
    l2 = np.sqrt((w * (f - h0) ** 2).sum() / (w * h0 ** 2).sum())
    M1 = (q[0] * dp[:, None, None] * ac[None, :, None]).sum()
    print(f"{case:7s} {scheme:>3s} K={K:<2d} hourly={int(hourly)}  L2 {l2:.4f}  max {f.max():.4f}  min {f.min():+.2e}  "
          f"dM/M0 {(M1 - M0) / M0:+.1e}  {el:.2f} s/step (last nsub {n})", flush=True)

for case in ('W1-a0', 'W1-a45', 'NL4'):
    run(case, 'lr')
    run(case, 'sl', 6, False)
    run(case, 'sl', 6, True)
    run(case, 'sl', 24, True)
    run(case, 'ffsl', 6, True)
    run(case, 'ffsl', 24, True)
