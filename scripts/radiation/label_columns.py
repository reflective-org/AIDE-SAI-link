#!/usr/bin/env python3
"""Label columns for the radiation emulator (docs/RADIATION_EMULATOR_SCOPE.md
§4.4): take saved 3-D aerosol states, pair each with a random archive hour
and per-column augmentations, run the EXACT physics on the full globe, and
keep a stratified sample of the columns with their features (rad_features.py)
and the raw physics outputs the trainer forms its targets from.

Per globe sample (one state x one hour):
  num   <- state bins x s_col,  s_col ~ log-uniform [S_MIN, S_MAX] per column;
           P_ZERO of the columns get s = 0 (clear sky: the structural zero),
           P_REF of them are replaced by the MAM4 reference aerosol x s_col
           (the background the reference call sees, for RAD_EMU_ANOM=net)
  dT    <- a state's dT_rad x u_col,  u_col ~ U[0, DT_MAX]; P_DT0 get 0
  three solves: evolved (num, T+dT), clear at T+dT, clear at T (cached)
Stored per kept column (float32): features (NF), raw (150):
  hr1_sw hr1_lw hceff_sw hceff_lw hc_sw hc_lw  [6 x nzb, K/day]
  swup1 olr1 swup_eff olr_eff swup_c olr_c     [W/m2]
and meta: state id, hour, lat/lon index, s_col, u_col, aod550, cosz, source
(0 evolved, 1 zero, 2 reference). Shards: <out>/g<NNNN>.npz.

Usage (GPU + archive, ~6 s per globe):
  CUDA_VISIBLE_DEVICES=N P_LO_HPA=1 P_HI_HPA=150 INIT_BIN=so4 python3 label_columns.py \
      --states a.npz b.npz ... --n-globes 200 --out DIR [--hours-min 2000] [--seed 0]
"""
import os, sys, time, glob, argparse
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--states', nargs='+', required=True, help='npz files with num (NBINS,nzb,nlat,nlon) and dT_rad')
ap.add_argument('--n-globes', type=int, default=100)
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--hours-min', type=int, default=2000, help='lowest archive hour to draw (hold out the first weeks)')
ap.add_argument('--hours-max', type=int, default=None)
ap.add_argument('--hours', default=None, help='fixed hour list a:b:step instead of random draws')
ap.add_argument('--keep', type=int, default=14000, help='target columns kept per globe')
ap.add_argument('--s-min', type=float, default=0.1); ap.add_argument('--s-max', type=float, default=10.0)
ap.add_argument('--p-zero', type=float, default=0.08); ap.add_argument('--p-ref', type=float, default=0.22)
ap.add_argument('--dt-max', type=float, default=2.5); ap.add_argument('--p-dt0', type=float, default=0.12)
ap.add_argument('--dt-cap', type=float, default=3.0, help='cap on |dT| after scaling [K] (the 1-year state reaches 12 K; a 30-day run 0.8 K)')
ap.add_argument('--no-aug', action='store_true', help='no scaling: the state as is, its own dT (validation sets)')
ap.add_argument('--start', type=int, default=0, help='first globe index (resume)')
ap.add_argument('--keep-all', action='store_true', help='keep every column (validation sets)')
ap.add_argument('--planck-only', action='store_true', help='zero aerosol: one clear solve at T+dT per globe (Planck-net coverage); optics features are zero, aerosol targets exactly zero')
ap.add_argument('--dt-noise', type=float, default=0.0, help='add N(0, S) K per level to dT (and give 30 %% of the columns a pure-noise profile): the sharp/zig-zag shapes an emulated run can produce')
args = ap.parse_args()

import coupling as C
import radiation as R
import rad_cache, rad_cache_build, rad_features as F
import jax, jax.numpy as jnp

rng = np.random.default_rng(args.seed)
os.makedirs(args.out, exist_ok=True)
t0 = time.time()
dU = C.open_var('U'); lat = dU['lat'].values; lon = dU['lon'].values
plev = dU['hyam'].values * dU['P0'].values + dU['hybm'].values * C.PS_REF
band = np.where((plev >= C.P_LO_HPA * 100) & (plev <= C.P_HI_HPA * 100))[0]
klevs = list(range(band.min(), band.max() + 1))
drv = R.RadiationDriver(C.open_var, klevs, lat, lon, C.DP_BIN * 1e-9, mmid_kg=C.MMID, rho_dry=C.RHO_AER)
drv.ref_cache = rad_cache.RefCache(C.RAD_REF_CACHE, drv.cache_key(**C.rad_ref_src()))
ds_mam = {f'{p}_a{m}': C.open_var(f'{p}_a{m}') for p in ('num', 'so4') for m in (1, 2, 3)}
nzb = len(klevs); nlat, nlon = lat.size, lon.size; ncol = nlat * nlon
ntime = drv.time.size
hmax = args.hours_max or (ntime - 2)
L = F.layout(drv.nzc, nzb, drv.nb_sw, drv.nb_lw)
print(f'driver ready {time.time()-t0:.0f}s; NF={L["NF"]} nzb={nzb}; cache {drv.ref_cache.dir}', flush=True)

states = []
for f in args.states:
    z = np.load(f)
    num = np.asarray(z['num'], np.float32)
    assert num.shape[1] == nzb, f'{f}: {num.shape} vs nzb {nzb}'
    dT = np.asarray(z['dT_rad'], np.float32)
    hour = int(z['hour_abs']) if 'hour_abs' in z.files else None
    states.append((f, num, dT, hour))
    print(f'  state {len(states)-1}: {f.split("/")[-2]}/{f.split("/")[-1]} |dT| max {np.abs(dT).max():.3f} hour {hour}', flush=True)
Cc = R.RAD_LAT_CHUNK
chunks = [(a, min(a + Cc, nlat)) for a in range(0, nlat, Cc)]
if args.hours:
    a, b, s = (int(x) for x in args.hours.split(':')); fixed_hours = list(range(a, b, s))
else:
    fixed_hours = None

for g in range(args.start, args.n_globes):
    tg = time.time()
    si = int(rng.integers(len(states)))
    f_state, num0, dT0, hour_state = states[si]
    if fixed_hours is not None:
        hour = fixed_hours[g % len(fixed_hours)]
    elif args.no_aug and hour_state is not None:
        hour = hour_state
    else:
        hour = int(rng.integers(args.hours_min, hmax))
    # ---- augmentations (per column) ----
    if args.no_aug:
        s_col = np.ones((nlat, nlon), np.float32); u_col = np.ones((nlat, nlon), np.float32)
        src = np.zeros((nlat, nlon), np.int8)
        num = num0; dT = dT0
    else:
        s_col = np.exp(rng.uniform(np.log(args.s_min), np.log(args.s_max), (nlat, nlon))).astype(np.float32)
        src = np.zeros((nlat, nlon), np.int8)
        r = rng.random((nlat, nlon))
        src[r < args.p_zero] = 1; src[(r >= args.p_zero) & (r < args.p_zero + args.p_ref)] = 2
        s_col[src == 1] = 0.0
        num = num0 * s_col[None, None]
        if (src == 2).any():
            num_ref = np.asarray(C.bin_mam4(ds_mam, hour - C.H0, klevs)[0], np.float32)
            num = np.where((src == 2)[None, None], num_ref * s_col[None, None], num)
        # dT from a random state (decorrelated from the aerosol), scaled per column
        dTs = states[int(rng.integers(len(states)))][2]
        u_col = rng.uniform(0.0, args.dt_max, (nlat, nlon)).astype(np.float32)
        u_col[rng.random((nlat, nlon)) < args.p_dt0] = 0.0
        dmax = np.abs(dTs).max(0)                              # per column
        u_col = np.minimum(u_col, args.dt_cap / np.maximum(dmax, 1e-6)).astype(np.float32)
        dT = dTs * u_col[None]
        if args.dt_noise > 0:
            pure = rng.random((nlat, nlon)) < 0.3
            dT = np.where(pure[None], 0.0, dT) + rng.normal(0.0, args.dt_noise, dT.shape).astype(np.float32) * (0.2 + 2.0 * rng.random((nlat, nlon)))[None].astype(np.float32)
    # ---- the physics ----
    h = F.prep_hour(drv, hour)
    T_band = h['T'][drv.band_pos]
    T_eff = T_band + dT.astype(np.float64)
    if args.planck_only:
        num = np.zeros_like(num); src[:] = 1
    hce, dce = drv.heating(hour, None, T_band_override=T_eff, split=True)
    if args.planck_only:
        hr1, d1 = hce, dce
    else:
        hr1, d1 = drv.heating(hour, jnp.asarray(num), T_band_override=T_eff, split=True)
    cl = rad_cache_build.ensure_clear(drv, hour)
    hr1 = np.asarray(hr1); hr1_sw = np.asarray(d1['hr_sw_band'])
    hce = np.asarray(hce); hce_sw = np.asarray(dce['hr_sw_band'])
    D = F.SEC_PER_DAY
    raw = np.concatenate([
        hr1_sw * D, (hr1 - hr1_sw) * D, hce_sw * D, (hce - hce_sw) * D,
        cl['hr_clear_sw'] * D, (cl['hr_clear'] - cl['hr_clear_sw']) * D,
        np.asarray(d1['sw_up_toa'])[None], np.asarray(d1['olr'])[None],
        np.asarray(dce['sw_up_toa'])[None], np.asarray(dce['olr'])[None],
        cl['dc_sw_up_toa'][None], cl['dc_olr'][None]], axis=0)        # (150, nlat, nlon)
    aod = np.asarray(d1['aod550'])
    # ---- features, chunk by chunk (same optics path as heating()) ----
    T_full = h['T'].copy(); T_full[drv.band_pos] = T_eff
    wwt = F.wt_weights_for(drv, T_full, h['q'])
    feats = []
    num_j = jnp.asarray(num)
    for a, b in chunks:
        nc = num_j[:, :, a:b]
        ww = jnp.asarray(wwt[:, :, a:b]) if wwt is not None else None
        tau_s, ssa_s, g_s, tau_l, ssa_l, g_l = F.optics_chunk(drv, nc, ww)
        fx = F.column_features(T_full[:, a:b], h['q'][:, a:b], h['o3'][:, a:b], h['ch4'][:, a:b],
                               h['ts'][a:b], h['alb'][a:b], h['zen'][a:b], dT[:, a:b],
                               tau_s, ssa_s, g_s, tau_l, ssa_l, g_l)
        feats.append(np.asarray(fx))
    feats = np.concatenate(feats, axis=0)                              # (ncol, NF)
    assert feats.shape == (ncol, L['NF'])
    # ---- stratified keep by column AOD ----
    aodf = aod.reshape(-1); srcf = src.reshape(-1)
    p = np.full(ncol, 0.12)
    p[aodf >= 1e-3] = 0.5; p[aodf >= 1e-2] = 1.0
    p[srcf == 1] = 0.3                      # clear columns: enough to pin the zero
    if args.planck_only:
        p[:] = 1.0; p[np.abs(dT).max(0).reshape(-1) < 0.02] = 0.3     # mostly the columns with a real dT
    keep = rng.random(ncol) < p
    idx = np.arange(ncol) if args.keep_all else np.nonzero(keep)[0]
    if idx.size > args.keep and not args.keep_all:
        idx = rng.choice(idx, args.keep, replace=False)
    idx.sort()
    yi, xi = np.unravel_index(idx, (nlat, nlon))
    meta = np.stack([np.full(idx.size, si), np.full(idx.size, hour), yi, xi,
                     s_col.reshape(-1)[idx], u_col.reshape(-1)[idx], aodf[idx],
                     np.maximum(np.cos(h['zen']), 0).reshape(-1)[idx], srcf[idx]], axis=1).astype(np.float64)
    np.savez(os.path.join(args.out, f'g{g:04d}.npz'),
             x=feats[idx], raw=raw.reshape(150, -1).T[idx].astype(np.float32), meta=meta,
             meta_keys=np.array(['state', 'hour', 'lat_i', 'lon_i', 's_col', 'u_col', 'aod550', 'cosz', 'src']),
             state_file=f_state, hour=hour, nf=L['NF'], nzb=nzb)
    da = raw[0:24] + raw[24:48] - raw[48:72] - raw[72:96]       # aerosol term K/day (sw+lw)
    pl = raw[48:72] + raw[72:96] - raw[96:120] - raw[120:144]   # Planck term
    print(f'  [{g+1}/{args.n_globes}] state {si} hour {hour} ({time.time()-tg:.1f}s): kept {idx.size} '
          f'(aod>=1e-2: {(aodf[idx]>=1e-2).sum()}, zero {(srcf[idx]==1).sum()}, ref {(srcf[idx]==2).sum()}); '
          f'aod max {aod.max():.3f}; aerosol term |.| p99 {np.percentile(np.abs(da),99):.4f} max {np.abs(da).max():.3f} K/d; '
          f'Planck term max {np.abs(pl).max():.3f} K/d; cache {drv.ref_cache.stats().split(" (")[0]}', flush=True)
print(f'LABEL_COLUMNS_DONE {args.n_globes} globes in {time.time()-t0:.0f}s', flush=True)
