#!/usr/bin/env python3
"""Transport-only run on the P2@200 emulator winds: age-of-air clock tracer +
passive injection tracer, using the campaign's advection operator unchanged
(src/advection/sl_lr.advect_step_batch, ADV_SCHEME=sl, ADV_SL_K=12, f32 gathers,
strang split, Jacobian air mass, global fixer, zonal polar caps, ADV_VPOS=1,
two step-end snapshots linear in time) called the way coupling.py calls it
(lat, dp=np.gradient(PLEV_PA), plev=PLEV_PA, lat_freeze=80, dt=6 h).

Tracer 0  CLOCK [years]: chi = t at the bottom band level (143 hPa) for |lat| <= 10
          (Dirichlet after every step, and the bottom-face inflow value there);
          elsewhere passive. Bottom/top-face inflow outside the source carries
          the local edge-cell value (zero-gradient). IC chi = 0 (age 0) everywhere.
          AOA = t - chi.
Tracer 1  INJ [kg/kg]: continuous point source, 5 Tg/yr into each of the two
          campaign cells (k nearest 55 hPa, lat nearest +-30, lon nearest 180),
          added at the step start like coupling.py's inj_dq. Inflow through both
          open faces = 0; the only sink is export through the band edges.
          Carried during the FIRST cycle only.

The 1997 met year (frames 0..1460) is cycled; step s uses frames s%1460, s%1460+1.
"""
import os, sys, time, json, argparse
os.environ.setdefault('ADV_SCHEME', 'sl'); os.environ.setdefault('ADV_SL_K', '12')
os.environ.setdefault('ADV_VPOS', '1'); os.environ.setdefault('ADV_F32', '1')
sys.path.insert(0, '/data/AIDE-SAI-link/src')
import _paths  # noqa: F401  (puts src/advection etc on sys.path)
import numpy as np
import jax, jax.numpy as jnp
import sl_lr

ap = argparse.ArgumentParser()
ap.add_argument('--years', type=int, default=6)
ap.add_argument('--out', default='/data/AIDE-SAI-link/linking_plots/climate_transport/data')
ap.add_argument('--met', default='/data/AIDE-SAI-link/runs/emu_winds_P2s200_1997/met')
ap.add_argument('--tag', default='aoa_run')
ap.add_argument('--mode', default='clock', choices=['clock', 'age'])
ap.add_argument('--no-inj', action='store_true')
ap.add_argument('--max-hours', type=float, default=2.75, help='wall-clock budget')
a = ap.parse_args()

MET = a.met
meta = json.load(open(f'{MET}/meta.json'))
lat = np.array(meta['lat']); lon = np.array(meta['lon']); PLEV = np.array(meta['plev_pa'])
DP = np.gradient(PLEV)
nlev, nlat, nlon = PLEV.size, lat.size, lon.size
STEP_SEC = 6 * 3600.0; NSTEP_YR = 1460; YR = 365.0
GRAV = 9.80616; RAD = 6.371e6; DEG = np.pi / 180

t0 = time.time()
print('loading winds into RAM ...', flush=True)
U = np.load(f'{MET}/u.npy'); V = np.load(f'{MET}/v.npy'); Wm = np.load(f'{MET}/omega.npy')
print(f'  loaded {U.shape} x3 in {time.time()-t0:.0f}s', flush=True)

trop = np.abs(lat) <= 10.0
trop_j = jnp.asarray(trop)
# injection increment (coupling.py _inj_increment, INJ_MIRROR=1, INJ_ZONAL=0)
k_inj = int(np.argmin(np.abs(PLEV / 100 - 55))); i_inj = int(np.argmin(np.abs(lon - 180)))
dlat = np.gradient(lat) * DEG; dlon = (lon[1] - lon[0]) * DEG
area = RAD**2 * np.cos(lat * DEG) * dlat * dlon
rate = 10e9 / (YR * 86400)
dq = np.zeros((nlev, nlat, nlon))
inj_cells = []
for L in (30, -30):
    j = int(np.argmin(np.abs(lat - L)))
    airm = DP[k_inj] / GRAV * area[j]
    dq[k_inj, j, i_inj] += 0.5 * rate * STEP_SEC / airm
    inj_cells.append((float(PLEV[k_inj] / 100), float(lat[j]), float(lon[i_inj]), float(airm)))
print('injection cells (hPa, lat, lon, air kg):', inj_cells, flush=True)
dq_j = jnp.asarray(dq)
# per-cell air mass [kg] with the model's own weight (exact cell-mean cos), for budgets
_e = np.concatenate([[lat[0] * DEG], 0.5 * (lat[:-1] + lat[1:]) * DEG, [lat[-1] * DEG]])
W_LAT = (np.sin(_e[1:]) - np.sin(_e[:-1])) / ((lat[1] - lat[0]) * DEG)
cell_area = RAD**2 * W_LAT * (lat[1] - lat[0]) * DEG * dlon       # m2 (exact band area)
AIRM = jnp.asarray(DP[:, None, None] / GRAV * cell_area[None, :, None] * np.ones((1, 1, nlon)))

nyears = a.years
q = jnp.zeros((2, nlev, nlat, nlon), jnp.float64)
akw = dict(lat=lat, dp=DP, lat_freeze=80.0, dt_total=STEP_SEC, plev=PLEV,
           dtype=jnp.float32, return_vflux=False)

daily_aoa_zm = []          # (ndays_total, nlev, nlat)
daily_inj_col = []         # year 1: (365, nlat, nlon) kg/m2
daily_inj_zm = []          # year 1: (365, nlev, nlat) kg/kg
snap = {}
budget = []                # (day, injected kg, in-band kg)
inj_cum = 0.0
tstep = []
s = 0
stop = False
for y in range(nyears):
    for si in range(NSTEP_YR):
        ts = time.time()
        i0 = si; i1 = si + 1
        u0, v0, w0 = U[i0], V[i0], Wm[i0]; u1, v1, w1 = U[i1], V[i1], Wm[i1]
        t_start = s * STEP_SEC / (YR * 86400); t_end = (s + 1) * STEP_SEC / (YR * 86400)
        if y == 0 and not a.no_inj:
            q = q.at[1].add(dq_j); inj_cum += float((dq_j * AIRM).sum())
            qb = q
        else:
            qb = q[:1]
        # inflow values at the open faces: clock = local edge value (zero-gradient),
        # t in the tropical source; inj = 0
        qfr = jnp.zeros_like(qb)
        qfr = qfr.at[0].set(qb[0])
        tmid = 0.5 * (t_start + t_end)
        if a.mode == 'clock':
            qfr = qfr.at[0, -1].set(jnp.where(trop_j[:, None], tmid, qb[0, -1]))
            out, nsub = sl_lr.advect_step_batch(qb, u0, v0, w0, u1, v1, w1, qfrozb=qfr, **akw)[:2]
            out = out.at[0, -1].set(jnp.where(trop_j[:, None], t_end, out[0, -1]))
        else:   # age tracer: advect, age by dt everywhere, reset to 0 in the source
            qfr = qfr.at[0, -1].set(jnp.where(trop_j[:, None], 0.0, qb[0, -1]))
            out, nsub = sl_lr.advect_step_batch(qb, u0, v0, w0, u1, v1, w1, qfrozb=qfr, **akw)[:2]
            out = out.at[0].add(t_end - t_start)
            out = out.at[0, -1].set(jnp.where(trop_j[:, None], 0.0, out[0, -1]))
        q = out if (y == 0 and not a.no_inj) else q.at[0].set(out[0]).at[1].set(0.0)
        s += 1
        if (s % 4) == 0:                       # daily diagnostics
            day = s // 4
            aoa = (t_end - q[0]) if a.mode == 'clock' else q[0]
            daily_aoa_zm.append(np.asarray(aoa.mean(-1), np.float32))
            if y == 0 and not a.no_inj:
                inj = q[1]
                col = (inj * (DP[:, None, None] / GRAV)).sum(0)
                daily_inj_col.append(np.asarray(col, np.float32))
                daily_inj_zm.append(np.asarray(inj.mean(-1), np.float32))
                budget.append((day, inj_cum, float((inj * AIRM).sum())))
                if day in (30, 90, 180, 365):
                    snap[f'inj3d_d{day}'] = np.asarray(inj, np.float32)
            if day % 30 == 0:
                print(f'  y{y} day {day} ({t_end:.3f} yr)  step {np.mean(tstep[-120:]):.3f}s  '
                      f'AOA trop143 {float(aoa[-1][trop].mean()):.3f}  AOA 10hPa eq '
                      f'{float(aoa[np.argmin(abs(PLEV-1000)), np.abs(lat)<5].mean()):.3f}  '
                      f'min/max AOA {float(aoa.min()):.3g}/{float(aoa.max()):.3g}'
                      + (f'  inj in-band {budget[-1][2]/max(inj_cum,1):.3f} of {inj_cum/1e9:.3f} Tg' if (y == 0 and budget) else ''),
                      flush=True)
        jax.block_until_ready(q)
        tstep.append(time.time() - ts)
        if time.time() - t0 > a.max_hours * 3600 and si == NSTEP_YR - 1:
            stop = True
    # end of year: checkpoint
    np.savez(f'{a.out}/{a.tag}.npz', aoa_zm=np.stack(daily_aoa_zm), inj_col=np.stack(daily_inj_col) if daily_inj_col else np.zeros(1),
             inj_zm=np.stack(daily_inj_zm) if daily_inj_zm else np.zeros(1), mode=a.mode, budget=np.array(budget), lat=lat, lon=lon, plev=PLEV,
             years_done=y + 1, step_time=np.array(tstep), inj_cells=np.array(inj_cells),
             clock_final=np.asarray(q[0], np.float32), t_final=s * STEP_SEC / (YR * 86400), **snap)
    print(f'== year {y+1} done, elapsed {(time.time()-t0)/60:.1f} min, mean step {np.mean(tstep):.3f}s', flush=True)
    # stop early if the next year would overrun the budget
    yr_sec = np.mean(tstep[-NSTEP_YR:]) * NSTEP_YR
    if time.time() - t0 + yr_sec > a.max_hours * 3600:
        print('budget reached, stopping', flush=True); break
print('DONE', flush=True)
