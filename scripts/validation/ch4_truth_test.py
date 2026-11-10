#!/usr/bin/env python3
"""Transport truth test against CESM's own CH4 (2026-11-05).

The only transported field in the h1 archive that is close to inert on a
one-year timescale in the band is CH4 (lifetime decades below ~30 hPa, a few
years at 1-10 hPa). CESM transported it with its own FV dycore (Lin-Rood FFSL
PPM on this grid, floating Lagrangian levels + remap). This script advects
CESM's CH4 field from the archive start with ONE of this model's transport
schemes on the same hourly winds, exactly as coupling.py calls it (6 h steps,
bracketing winds, open faces fed by the truth, polar caps stirred, f32), and
scores it against CESM's CH4 at each step. It is the first transport test
whose answer does not come from fct_lr.

    CUDA_VISIBLE_DEVICES=2 python3 scripts/validation/ch4_truth_test.py --scheme ffsl --k 24 --days 365 --tag ffslX_K24

Writes outputs/ch4_truth/<tag>.npz with the per-step scores and the final
zonal means, and prints a table every --report days. Scores (RMSE, bias,
pattern correlation) are over |lat| <= 80 and p >= P_SCORE_HPA (default 30),
area- and dp-weighted; 'persist' is the do-nothing baseline (IC held fixed).
"""
import argparse, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src')); import _paths  # noqa
import jax, jax.numpy as jnp
import xarray as xr

ap = argparse.ArgumentParser()
ap.add_argument('--scheme', default='ffsl', choices=['lr', 'ffsl', 'sl'])
ap.add_argument('--k', type=int, default=None, help='sub-steps (ffsl/sl)')
ap.add_argument('--frac', default=None, help='ffsl sliver measure cell|face|exact')
ap.add_argument('--split', default=None, help='ffsl strang|lie')
ap.add_argument('--cfl', type=float, default=0.5)
ap.add_argument('--days', type=int, default=365)
ap.add_argument('--report', type=int, default=10)
ap.add_argument('--p-lo', type=float, default=1.0); ap.add_argument('--p-hi', type=float, default=150.0)
ap.add_argument('--p-score', type=float, default=30.0, help='score only levels with p >= this [hPa]')
ap.add_argument('--hourly', action='store_true', help='pass the hourly wind snapshots within the step (ADV_SL_HOURLY=1 path)')
ap.add_argument('--tag', required=True)
args = ap.parse_args()

if args.frac: os.environ['ADV_FFSL_FRAC'] = args.frac
if args.split: os.environ['ADV_FFSL_SPLIT'] = args.split
if args.k and args.scheme == 'ffsl': os.environ['ADV_FFSL_K'] = str(args.k)
if args.k and args.scheme == 'sl': os.environ['ADV_SL_K'] = str(args.k)
if args.scheme == 'lr':
    import fct_lr as M
elif args.scheme == 'ffsl':
    import ffsl_lr as M
else:
    import sl_lr as M
desc = M.describe() if hasattr(M, 'describe') else f'fct_lr cfl={args.cfl}'
print(f"scheme {args.scheme}: {desc}", flush=True)

HDIR = ('/data/cesm2.1.5_output/histSST/f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001/'
        'archive/atm/proc/tseries/hour_1/f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001.cam.h1.')
SUF = '.1996010100-2014123100.nc'
ds = {v: xr.open_dataset(HDIR + v + SUF, decode_times=False) for v in ('U', 'V', 'OMEGA', 'CH4')}
lat = ds['U']['lat'].values; lon = ds['U']['lon'].values
plev = ds['U']['hyam'].values * ds['U']['P0'].values + ds['U']['hybm'].values * 1.0e5
band = np.where((plev >= args.p_lo * 100) & (plev <= args.p_hi * 100))[0]
klevs = list(range(band.min(), band.max() + 1))
PLEV = plev[klevs]; DP = np.gradient(PLEV); nlev = len(klevs)
print(f"grid {lat.size}x{lon.size}, {nlev} levels ({PLEV[0]/100:.1f}..{PLEV[-1]/100:.1f} hPa)", flush=True)

def rd(v, t):
    return ds[v][v].isel(time=t, lev=klevs).values
def winds(t):
    return rd('U', t), rd('V', t), rd('OMEGA', t)
def ch4(t):
    return rd('CH4', t)

STEP_H = 6; STEP_SEC = STEP_H * 3600.0
nsteps = args.days * 24 // STEP_H
# scoring weights: area (cos lat) x dp, |lat| <= 80, p >= p_score
coslat = np.cos(np.deg2rad(lat)); coslat[np.abs(lat) > 80] = 0.0
lv = (PLEV >= args.p_score * 100).astype(float)
W = (lv[:, None] * DP[:, None] * coslat[None, :])[:, :, None] * np.ones((1, 1, lon.size))
W /= W.sum()
def score(x, y):
    d = x - y
    rmse = np.sqrt((W * d * d).sum()); bias = (W * d).sum()
    xm = (W * x).sum(); ym = (W * y).sum()
    cov = (W * (x - xm) * (y - ym)).sum(); sx = np.sqrt((W * (x - xm) ** 2).sum()); sy = np.sqrt((W * (y - ym) ** 2).sum())
    return rmse, bias, cov / (sx * sy), sy
def zm(x):
    return x.mean(-1)

q0 = ch4(0)
q = jnp.asarray(q0[None], dtype=jnp.float64)       # (1, nlev, nlat, nlon)
qfroz = jnp.asarray(q0[None], dtype=jnp.float64)
u0, v0, w0 = winds(0)
akw = dict(lat=lat, dp=DP, lat_freeze=80.0, dt_total=STEP_SEC, cfl=args.cfl, dtype=jnp.float32)
if args.scheme == 'sl':
    akw['plev'] = PLEV
rows = []; t_adv = []
t_start = time.time()
for s in range(nsteps):
    it1 = (s + 1) * STEP_H
    u1, v1, w1 = winds(it1)
    if args.hourly:
        akw['hourly_winds'] = [(u0, v0, w0)] + [winds(t) for t in range(it1 - STEP_H + 1, it1)] + [(u1, v1, w1)]
    ta = time.time()
    out = M.advect_step_batch(q, u0, v0, w0, u1, v1, w1, qfrozb=qfroz, **akw)
    q = out[0]; nsub = out[1]
    q.block_until_ready(); t_adv.append(time.time() - ta)
    truth = ch4(it1)
    qfroz = jnp.asarray(truth[None], dtype=jnp.float64)       # reservoir / caps = truth at the step end
    qn = np.asarray(q[0])
    r_m = score(qn, truth); r_p = score(q0, truth)
    rows.append((it1 / 24.0, nsub) + r_m + r_p[:3])
    u0, v0, w0 = u1, v1, w1
    if (s + 1) % (args.report * 24 // STEP_H) == 0 or s == nsteps - 1:
        day = it1 / 24
        print(f"day {day:6.1f}  nsub {nsub:4d}  advect {np.median(t_adv[-10:]):.2f}s  |  model: rmse {r_m[0]:.3e} bias {r_m[1]:+.3e} corr {r_m[2]:.4f}"
              f"  |  persist: rmse {r_p[0]:.3e} bias {r_p[1]:+.3e} corr {r_p[2]:.4f}  |  truth std {r_m[3]:.3e}"
              f"  |  rmse ratio model/persist {r_m[0]/r_p[0]:.3f}", flush=True)
rows = np.array(rows)
os.makedirs(f'{ROOT}/outputs/ch4_truth', exist_ok=True)
np.savez(f'{ROOT}/outputs/ch4_truth/{args.tag}.npz', rows=rows,
         cols=np.array(['day', 'nsub', 'rmse', 'bias', 'corr', 'truth_std', 'p_rmse', 'p_bias', 'p_corr']),
         zm_model=zm(np.asarray(q[0])), zm_truth=zm(truth), zm_ic=zm(q0), plev=PLEV, lat=lat,
         scheme=args.scheme, desc=desc, t_adv_median=float(np.median(t_adv)), wall=time.time() - t_start)
print(f"done {nsteps} steps in {time.time()-t_start:.0f}s; advect median {np.median(t_adv):.3f}s/step; wrote outputs/ch4_truth/{args.tag}.npz", flush=True)
