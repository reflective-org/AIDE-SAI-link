#!/usr/bin/env python3
"""Time dinosaur's semi-Lagrangian tracer transport at the AIDE-SAI-link grid size.

82 tracers (40 number + 40 mass + SO2 + H2SO4) x 24 levels x 192 lat x 288 lon,
one 6-hour step, K sub-steps, cubic Lagrange gather + quasi-monotone limiter
(the JCM configuration), on one GPU.  Horizontal departure points per level
(2 fixed-point iterations) + 3-D gather with linear vertical interpolation at
a prescribed vertical displacement.  Synthetic winds; this is a COST
measurement, not an accuracy one.
"""
import os, sys, time
sys.path.insert(0, '/data/dinosaur-sl')
import numpy as np
import jax, jax.numpy as jnp
from dinosaur import spherical_harmonic as sh, semi_lagrangian as sl

print('jax', jax.__version__, jax.devices(), flush=True)
NLON, NLAT, NLEV, NTR = 288, 192, 24, 82
DT6 = 21600.0
# A Gaussian grid with the model's node counts (dinosaur does not support nodes
# AT the poles, which the CESM 0.9x1.25 grid has -- see the writeup).
grid = sh.Grid(longitude_wavenumbers=NLON // 2, total_wavenumbers=NLON // 2 + 1,
               longitude_nodes=NLON, latitude_nodes=NLAT, radius=6.371e6)
lon_m, sinlat_m = grid.nodal_mesh
print('grid nodal_shape', grid.nodal_shape, 'lat range', float(np.degrees(np.arcsin(np.asarray(sinlat_m)[0]).min())),
      float(np.degrees(np.arcsin(np.asarray(sinlat_m)[0]).max())), flush=True)

rng = np.random.default_rng(0)
lat2 = np.arcsin(np.asarray(sinlat_m))                          # (lon, lat)
# stratospheric-ish winds: 40 m/s zonal jet + 10 m/s eddies, per level
u = (40.0 * np.cos(lat2)[None] + 10.0 * rng.standard_normal((NLEV, NLON, NLAT))).astype(np.float32)
v = (10.0 * rng.standard_normal((NLEV, NLON, NLAT))).astype(np.float32)
q = np.exp(rng.standard_normal((NTR, NLEV, NLON, NLAT))).astype(np.float32)
sigma_nodes = np.linspace(0.02, 0.98, NLEV)                      # fixed level coordinate
sig_arr = np.broadcast_to(sigma_nodes[:, None, None], (NLEV, NLON, NLAT))
dsig = (0.05 * rng.standard_normal((NLEV, NLON, NLAT)) * (sigma_nodes[1] - sigma_nodes[0])).astype(np.float32)

u, v, q = jnp.asarray(u), jnp.asarray(v), jnp.asarray(q)
sig_dep = jnp.clip(jnp.asarray(sig_arr + dsig, dtype=jnp.float32), sigma_nodes[0], sigma_nodes[-1])

interp = sl.GridInterpolator(grid, 'cubic', 'quasi_monotone')

def step_2d(q, u, v, K):
    dt = DT6 / K
    for k in range(K):
        dep = sl.horizontal_departure_points(u, v, grid, dt, iterations=2)
        q = jax.vmap(lambda f: sl.transport_scalar_2d(f, dep, interp))(q)
    return q

def step_3d(q, u, v, K):
    dt = DT6 / K
    for k in range(K):
        dep = sl.horizontal_departure_points(u, v, grid, dt, iterations=2)
        lon_d, sl_d = dep.lon, dep.sin_lat
        q = jax.vmap(lambda f: sl.interpolate_3d(f, grid, lon_d, sl_d, sig_dep, sigma_nodes,
                                                 order='cubic', limiter='quasi_monotone',
                                                 vertical_order='linear'))(q)
    return q

def bench(name, fn, K, n=10):
    f = jax.jit(lambda q, u, v: fn(q, u, v, K))
    t0 = time.time(); out = f(q, u, v); out.block_until_ready(); tc = time.time() - t0
    ts = []
    for _ in range(n):
        t0 = time.time(); out = f(q, u, v); out.block_until_ready(); ts.append(time.time() - t0)
    ms = 1e3 * np.median(ts)
    print(f'{name:28s} K={K:2d}  {ms:8.2f} ms / 6 h step  (compile {tc:.1f}s, {ms/NTR:.3f} ms per tracer, '
          f'{1e6*np.median(ts)/(NTR*NLEV*NLON*NLAT):.4f} us per tracer-cell)', flush=True)
    return float(np.median(ts))

res = {}
for K in (1, 6):
    res[f'2d_K{K}'] = bench('horizontal SL (2-D gather)', step_2d, K)
    res[f'3d_K{K}'] = bench('3-D SL (cubic h, linear v)', step_3d, K)
# the pure gather cost: departure points once, 82 gathers
dep = jax.jit(lambda u, v: sl.horizontal_departure_points(u, v, grid, DT6, iterations=2))
d = dep(u, v); d.cartesian.block_until_ready()
t0 = time.time(); d = dep(u, v); d.cartesian.block_until_ready(); print(f'departure points alone (24 levels): {1e3*(time.time()-t0):.2f} ms', flush=True)
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sl_bench_result.npz'), **res)
print('SL_BENCH_DONE', res, flush=True)
