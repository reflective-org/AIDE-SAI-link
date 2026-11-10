#!/usr/bin/env python3
"""Passive injection tracer, two years (1997-1998) of P2@200 emulator winds, transport only.

Same tracer as the year-1 one in aoa_transport.py (tracer 1): continuous point source, 5 Tg/yr
into each of the two campaign cells (level nearest 55 hPa, lat nearest +-30, lon nearest 180),
added at the step start; inflow through both open faces = 0; the only sink is export through the
band edges. Campaign operator unchanged (sl, K=12, f32, strang, global fixer = campaign default,
which is right for a mass tracer). Met = the top 24 levels of the full-column 1997-98 met
(identical to the band met). Output: daily column mass and zonal mean.
"""
import os, sys, time, json, argparse
os.environ.setdefault('ADV_SCHEME', 'sl'); os.environ.setdefault('ADV_SL_K', '12')
os.environ.setdefault('ADV_VPOS', '1'); os.environ.setdefault('ADV_F32', '1')
sys.path.insert(0, '/data/AIDE-SAI-link/src')
import _paths  # noqa: F401
import numpy as np
import jax, jax.numpy as jnp
import sl_lr

ap = argparse.ArgumentParser()
ap.add_argument('--met', default='/raid/aide_sai_scratch/met_fullcol_1997_1998')
ap.add_argument('--nlev', type=int, default=24)
ap.add_argument('--out', default='/data/AIDE-SAI-link/linking_plots/climate_transport/data/inj2yr.npz')
a = ap.parse_args()
meta = json.load(open(f'{a.met}/meta.json'))
lat = np.array(meta['lat']); lon = np.array(meta['lon']); PLEV = np.array(meta['plev_pa'])[:a.nlev]
DP = np.gradient(PLEV); nlev, nlat, nlon = PLEV.size, lat.size, lon.size
STEP_SEC = 6 * 3600.0; YR = 365.0; NSTEP = 2920; GRAV = 9.80616; RAD = 6.371e6; DEG = np.pi / 180
t0 = time.time()
U, V, W = (np.ascontiguousarray(np.load(f'{a.met}/{k}.npy', mmap_mode='r')[:NSTEP + 1, :a.nlev]) for k in ('u', 'v', 'omega'))
print(f'loaded {U.shape} x3 in {time.time() - t0:.0f}s, plev {PLEV[0]/100:.2f}-{PLEV[-1]/100:.1f} hPa, fixer {os.environ.get("ADV_SL_FIXER", "global (default)")}', flush=True)

k_inj = int(np.argmin(np.abs(PLEV / 100 - 55))); i_inj = int(np.argmin(np.abs(lon - 180)))
dlat = np.gradient(lat) * DEG; dlon = (lon[1] - lon[0]) * DEG
area = RAD**2 * np.cos(lat * DEG) * dlat * dlon
rate = 10e9 / (YR * 86400)
dq = np.zeros((nlev, nlat, nlon)); cells = []
for L in (30, -30):
    j = int(np.argmin(np.abs(lat - L))); airm = DP[k_inj] / GRAV * area[j]
    dq[k_inj, j, i_inj] += 0.5 * rate * STEP_SEC / airm; cells.append((PLEV[k_inj] / 100, lat[j], lon[i_inj]))
print('injection cells (hPa, lat, lon):', cells, flush=True)
dq_j = jnp.asarray(dq)
_e = np.concatenate([[lat[0] * DEG], 0.5 * (lat[:-1] + lat[1:]) * DEG, [lat[-1] * DEG]])
W_LAT = (np.sin(_e[1:]) - np.sin(_e[:-1])) / ((lat[1] - lat[0]) * DEG)
cell_area = RAD**2 * W_LAT * (lat[1] - lat[0]) * DEG * dlon
AIRM = jnp.asarray(DP[:, None, None] / GRAV * cell_area[None, :, None] * np.ones((1, 1, nlon)))
DPG = jnp.asarray(DP[:, None, None] / GRAV)

akw = dict(lat=lat, dp=DP, lat_freeze=80.0, dt_total=STEP_SEC, plev=PLEV, dtype=jnp.float32, return_vflux=False)
q = jnp.zeros((1, nlev, nlat, nlon), jnp.float64); qfr = jnp.zeros_like(q)
col, zm, budget, tstep = [], [], [], []; inj_cum = 0.0
for s in range(NSTEP):
    ts = time.time()
    q = q.at[0].add(dq_j); inj_cum += float((dq_j * AIRM).sum())
    q = sl_lr.advect_step_batch(q, U[s], V[s], W[s], U[s + 1], V[s + 1], W[s + 1], qfrozb=qfr, **akw)[0]
    if (s + 1) % 4 == 0:
        x = q[0]
        col.append(np.asarray((x * DPG).sum(0), np.float32)); zm.append(np.asarray(x.mean(-1), np.float32))
        budget.append(((s + 1) // 4, inj_cum, float((x * AIRM).sum())))
        if ((s + 1) // 4) % 30 == 0:
            print(f'day {(s+1)//4}  step {np.mean(tstep[-120:]):.3f}s  in-band {budget[-1][2]/1e9:.3f} of {inj_cum/1e9:.3f} Tg', flush=True)
    jax.block_until_ready(q); tstep.append(time.time() - ts)
np.savez(a.out, inj_col=np.stack(col), inj_zm=np.stack(zm), budget=np.array(budget), lat=lat, lon=lon, plev=PLEV,
         inj_cells=np.array(cells), step_time=np.array(tstep))
print(f'DONE {a.out}  {(time.time()-t0)/60:.1f} min, mean step {np.mean(tstep):.3f}s', flush=True)
