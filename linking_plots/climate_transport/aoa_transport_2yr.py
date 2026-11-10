#!/usr/bin/env python3
"""Age of air on the 1997-1998 emulator winds, cycled (2026-11-09). A COPY of aoa_transport.py --mode age, generalised:
  * the met (band or full column) is cycled over its 2920 six-hour steps (1997-01-01T06 .. 1999-01-01T06, then wraps)
  * reset modes (all age tracers: advect, add dt everywhere, set 0 in the reset region; inflow at the bottom face = 0
    where the reset region touches it, else the local edge value; top face zero-gradient):
      trop143   band met: bottom band level (143 hPa), |lat| <= 10   (= the published setup)
      p150      full-column met: every cell with p >= 150 hPa      (jcm-strat 'aoa150' entry age)
      sfc       full-column met: every cell at or below the local surface (p >= ps(t)) and the lowest level
                (jcm-strat 'aoa_sfc'; comparable to the CLaMS surface clock)
Operator: src/advection/sl_lr.advect_step_batch exactly as coupling.py calls it, ADV_SCHEME=sl K=12 f32, with
ADV_SL_FIXER=none (the global fixer corrupts an age tracer, see NOTES.md).
Output: data/<tag>.npz with daily zonal-mean age per tracer, (ndays, nlev, nlat), averaged over ABOVE-GROUND cells only.
usage: aoa_transport_2yr.py --met DIR --resets trop143 | p150,sfc --years 10 --tag TAG"""
import os, sys, time, json, argparse
os.environ.setdefault('ADV_SCHEME', 'sl'); os.environ.setdefault('ADV_SL_K', '12')
os.environ.setdefault('ADV_VPOS', '1'); os.environ.setdefault('ADV_F32', '1')
os.environ['ADV_SL_FIXER'] = 'none'
sys.path.insert(0, '/data/AIDE-SAI-link/src')
import _paths  # noqa: F401
import numpy as np
import jax, jax.numpy as jnp
import sl_lr

ap = argparse.ArgumentParser()
ap.add_argument('--met', required=True)
ap.add_argument('--resets', required=True)
ap.add_argument("--years", type=int, default=5, help="number of 2-YEAR CYCLES of the 1997-1998 winds (5 = 10 model years)")
ap.add_argument('--tag', required=True)
ap.add_argument('--nlev', type=int, default=0, help='use only the top NLEV levels (24 = the band; identical to the band met)')
ap.add_argument('--out', default='/data/AIDE-SAI-link/linking_plots/climate_transport/data')
a = ap.parse_args()
R = a.resets.split(',')
meta = json.load(open(f'{a.met}/meta.json'))
lat = np.array(meta['lat']); lon = np.array(meta['lon']); PLEV = np.array(meta['plev_pa'])
NL = a.nlev or PLEV.size; PLEV = PLEV[:NL]; DP = np.gradient(PLEV)
nlev, nlat, nlon = PLEV.size, lat.size, lon.size
STEP_SEC = 6 * 3600.0; YR = 365.0; NSTEP = 2920
t0 = time.time()
U, V, W = (np.ascontiguousarray(np.load(f'{a.met}/{k}.npy', mmap_mode='r')[:, :NL]) for k in ('u', 'v', 'omega'))
PS = np.load(f'{a.met}/ps.npy') if os.path.exists(f'{a.met}/ps.npy') and PLEV.max() > 5e4 else None
assert U.shape[0] >= NSTEP + 1, U.shape
print(f'loaded {U.shape} x3 in {time.time() - t0:.0f}s; resets {R}; fixer {os.environ["ADV_SL_FIXER"]}', flush=True)
trop = jnp.asarray(np.abs(lat) <= 10.0)
P3 = jnp.asarray(PLEV)[:, None, None]


def reset_mask(r, i):
    """bool (nlev, nlat, nlon): cells set to age 0 at met frame i."""
    if r == 'trop143':
        m = jnp.zeros((nlev, nlat, nlon), bool); return m.at[-1].set(jnp.broadcast_to(trop[:, None], (nlat, nlon)))
    if r == 'p150':
        return jnp.broadcast_to(P3 >= 150e2, (nlev, nlat, nlon))
    if r == 'sfc':
        m = P3 >= jnp.asarray(PS[i])[None]
        return m.at[-1].set(True)
    raise ValueError(r)


akw = dict(lat=lat, dp=DP, lat_freeze=80.0, dt_total=STEP_SEC, plev=PLEV, dtype=jnp.float32, return_vflux=False)
nt = len(R)
q = jnp.zeros((nt, nlev, nlat, nlon), jnp.float64)
daily = []; tstep = []; s = 0
dt_yr = STEP_SEC / (YR * 86400)
for y in range(a.years):
    for si in range(NSTEP):
        ts = time.time()
        i0, i1 = si, si + 1
        M0 = jnp.stack([reset_mask(r, i0) for r in R]); M1 = jnp.stack([reset_mask(r, i1) for r in R])
        qfr = jnp.where(M0[:, -1:], 0.0, q[:, -1:])                       # bottom-face inflow value
        qfr = jnp.concatenate([q[:, :-1], qfr], axis=1)
        out = sl_lr.advect_step_batch(q, U[i0], V[i0], W[i0], U[i1], V[i1], W[i1], qfrozb=qfr, **akw)[0]
        out = out + dt_yr
        q = jnp.where(M1, 0.0, out)
        s += 1
        if s % 4 == 0:
            above = (P3 < jnp.asarray(PS[i1])[None]) if PS is not None else jnp.ones((nlev, nlat, nlon), bool)
            cnt = above.sum(-1)
            zm = jnp.where(above[None], q, 0.0).sum(-1) / jnp.where(cnt > 0, cnt, 1)[None]
            daily.append(np.asarray(jnp.where(cnt[None] > 0, zm, jnp.nan), np.float32))
            if (s // 4) % 60 == 0:
                k10 = int(np.argmin(abs(PLEV - 1000))); eq = np.abs(lat) < 5
                print(f'  y{y} day {s // 4} ({s * dt_yr:.2f} yr) step {np.mean(tstep[-200:]):.3f}s  ' +
                      '  '.join(f'{r}: 10hPa eq {float(np.nanmean(daily[-1][n, k10][eq])):.3f} max {float(np.nanmax(daily[-1][n])):.2f}' for n, r in enumerate(R)), flush=True)
        jax.block_until_ready(q)
        tstep.append(time.time() - ts)
    np.savez(f'{a.out}/{a.tag}.npz', aoa_zm=np.stack(daily), resets=np.array(R), lat=lat, plev=PLEV, years_done=y + 1,
             t_final=s * dt_yr, step_time=np.array(tstep[-NSTEP:]), met=a.met,
             age_final=np.asarray(q, np.float32), note='daily zonal mean over above-ground cells, years; cycled 1997-1998 winds')
    print(f'== year {y + 1} done, elapsed {(time.time() - t0) / 60:.1f} min, mean step {np.mean(tstep[-NSTEP:]):.3f}s', flush=True)
print('DONE', flush=True)
