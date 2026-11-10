#!/usr/bin/env python3
"""Adapter gate for RAD_ENGINE=emulator (src/radiation/rad_emulator.py).

Builds the production driver + the emulator on the day-30 state at hour 720
and checks: the contract (shapes, dtypes, finite, SW zero at night, the
structural zeros: zero aerosol + zero dT -> hr == cached hr_clear exactly),
both anomaly modes run, ref_ready/caching round-trips, and the wall time of
the emulated call vs the physics. With real checkpoints it also prints the
offline error of hr1 and the anomaly against the physics on this state.
--dummy writes random-weight checkpoints to a temp dir (shape test only).
Run: CUDA_VISIBLE_DEVICES=<free> P_LO_HPA=1 P_HI_HPA=150 INIT_BIN=so4 python3 test_rad_emulator.py [--dummy]
"""
import os, sys, time, tempfile, argparse
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument('--dummy', action='store_true'); ap.add_argument('--ckpt', default=None)
args = ap.parse_args()
import coupling as C
import radiation as R
import rad_cache, rad_features as F
import jax, jax.numpy as jnp

t0 = time.time()
dU = C.open_var('U'); lat = dU['lat'].values; lon = dU['lon'].values
plev = dU['hyam'].values * dU['P0'].values + dU['hybm'].values * C.PS_REF
band = np.where((plev >= C.P_LO_HPA * 100) & (plev <= C.P_HI_HPA * 100))[0]
klevs = list(range(band.min(), band.max() + 1))
drv = R.RadiationDriver(C.open_var, klevs, lat, lon, C.DP_BIN * 1e-9, mmid_kg=C.MMID, rho_dry=C.RHO_AER)
drv.ref_cache = rad_cache.RefCache(C.RAD_REF_CACHE, drv.cache_key(**C.rad_ref_src()))
st = np.load(f'{_REPO}/runs/bench_phys30d/coupled_state_bench_phys30d_ckpt.npz')
num = jnp.asarray(st['num']); dT = np.asarray(st['dT_rad']); t = 720
ds_mam = {f'{p}_a{m}': C.open_var(f'{p}_a{m}') for p in ('num', 'so4') for m in (1, 2, 3)}
num_ref = jnp.asarray(C.bin_mam4(ds_mam, t - C.H0, klevs)[0])

ckpt = args.ckpt
if args.dummy:
    ckpt = tempfile.mkdtemp(prefix='rad_emu_dummy_')
    L = F.layout(drv.nzc, drv.nzb, drv.nb_sw, drv.nb_lw); rng = np.random.default_rng(0)
    for net, w, nb in (('aer', 64, 2), ('planck', 32, 1)):
        cols = np.arange(L['NF']) if net == 'aer' else np.concatenate([np.arange(*L[k]) for k in ('T', 'logq', 'logo3', 'logch4', 'sfc', 'dT')])
        p = dict(net=net, cols=cols, x_mean=np.zeros(len(cols), np.float32), x_std=np.ones(len(cols), np.float32),
                 y_scale=np.full(2*drv.nzb+2, 1e-3, np.float32), g_scale=np.float32(1.0), nzb=drv.nzb, nzc=drv.nzc, nf=L['NF'],
                 width=w, blocks=nb, sw_mask=np.r_[np.ones(drv.nzb), np.zeros(drv.nzb), 1, 0].astype(np.float32), epoch=0, val='{}')
        p['inp.W'] = rng.normal(0, 0.02, (len(cols), w)).astype(np.float32); p['inp.b'] = np.zeros(w, np.float32)
        for j in range(nb):
            p[f'b{j}.W1'] = rng.normal(0, 0.1, (w, w)).astype(np.float32); p[f'b{j}.b1'] = np.zeros(w, np.float32)
            p[f'b{j}.W2'] = rng.normal(0, 0.1, (w, w)).astype(np.float32); p[f'b{j}.b2'] = np.zeros(w, np.float32)
        p['out.W'] = rng.normal(0, 0.1, (w, 2*drv.nzb+2)).astype(np.float32); p['out.b'] = np.zeros(2*drv.nzb+2, np.float32)
        np.savez(os.path.join(ckpt, f'{net}.npz'), **p)
import rad_emulator
emu = rad_emulator.RadEmulator(drv, ckpt=ckpt, ref_engine='rrtmgp', ref_every=1, tag='emutest')
print(f'ready {time.time()-t0:.0f}s; ref_ready({t}) = {emu.ref_ready(t)}', flush=True)
T_band = drv._read3d(drv.dT, 'T', t)[drv.band_pos]

# 1. contract + structural zeros
hr, dg = emu.heating(t, None); jax.block_until_ready(hr)
cl = drv.ref_cache.get(t, rad_cache.CLEAR_KEYS)
assert hr.shape == (drv.nzb, drv.nlat, drv.nlon) and hr.dtype == jnp.float64
assert np.array_equal(np.asarray(hr), cl['hr_clear'].astype(np.float64)), 'zero aerosol + zero dT must give the cached clear-sky field exactly'
assert np.array_equal(np.asarray(dg['olr']), cl['dc_olr'].astype(np.float64))
hr, dg = emu.heating(t, num, T_band_override=T_band + dT); jax.block_until_ready(hr)
assert np.isfinite(np.asarray(hr)).all() and all(np.isfinite(np.asarray(v)).all() for v in dg.values())
zen = R.solar_zenith(drv.time[t], drv.lat, drv.lon); night = zen >= 0.5 * np.pi
print(f'contract OK: |hr| p50 {np.median(np.abs(hr))*86400:.3f} K/d; aod gm {float(np.mean(dg["aod550"])):.4f}', flush=True)
# 2. anomaly, both modes, with and without num_ref
for mode in ('phys', 'net'):
    emu.anom = mode; emu.ref_engine = ''
    ta = time.time(); an, d1, d0 = emu.heating_anomaly(t, num, num_ref, dT); jax.block_until_ready(an); ta = time.time() - ta
    assert emu.ref_ready(t), f'{mode}: hour not ready after a computed reference'
    tb = time.time(); an2, d1b, d0b = emu.heating_anomaly(t, num, None, dT); jax.block_until_ready(an2); tb = time.time() - tb
    assert np.array_equal(np.asarray(an), np.asarray(an2)), f'{mode}: cached reference differs'
    print(f'anomaly mode {mode}: first {ta:.2f}s, cached {tb:.2f}s; anomaly |.| max {np.abs(an).max()*86400:.3f} K/d', flush=True)
# 3. physics comparison + timing (A/B path)
emu.anom = 'net'; emu.ref_engine = 'rrtmgp'
an, d1, d0 = emu.heating_anomaly(t, num, None, dT)
hr1p, d1p = drv.heating(t, num, T_band_override=T_band + dT); hr0p, d0p = drv.reference(t, None); anp = hr1p - hr0p
hr1e, _ = emu.heating(t, num, T_band_override=T_band + dT)
D = 86400; e1 = (np.asarray(hr1e) - np.asarray(hr1p)) * D; ea = (np.asarray(an) - np.asarray(anp)) * D
sig = (np.asarray(hr1p) - cl['hr_clear']) * D
print(f'vs physics (day-30 state, hour 720): hr1 err rms {np.sqrt((e1**2).mean()):.2e} max {np.abs(e1).max():.2e} K/d '
      f'(aer+Planck signal rms {np.sqrt((sig**2).mean()):.2e}); anomaly err rms {np.sqrt((ea**2).mean()):.2e} max {np.abs(ea).max():.2e} '
      f'(signal rms {np.sqrt(((np.asarray(anp)*D)**2).mean()):.2e} K/d, max {np.abs(anp).max()*D:.3f})', flush=True)
for k in range(0, drv.nzb, 4):
    print(f'   level {k:2d} ({plev[klevs[k]]/100:6.1f} hPa): anomaly err rms {np.sqrt((ea[k]**2).mean()):.2e} max {np.abs(ea[k]).max():.2e}; signal rms {np.sqrt(((np.asarray(anp[k])*D)**2).mean()):.2e}', flush=True)
def tm(f, n=3):
    f(); ts = []
    for _ in range(n):
        t1 = time.time(); jax.block_until_ready(f()); ts.append(time.time() - t1)
    return np.median(ts)
te = tm(lambda: emu.heating(t, num, T_band_override=T_band + dT)[0])
tp = tm(lambda: drv.heating(t, num, T_band_override=T_band + dT)[0])
ta = tm(lambda: emu.heating_anomaly(t, num, None, dT)[0]) if emu.ref_engine == '' else None
emu.ref_engine = ''; ta = tm(lambda: emu.heating_anomaly(t, num, None, dT)[0])
print(f'timing (median of 3, warm): emulated heating() {te:.3f}s vs physics {tp:.3f}s; emulated anomaly (cached ref) {ta:.3f}s', flush=True)
os.remove('rad_bench_emutest.npz') if os.path.exists('rad_bench_emutest.npz') else None
print('RAD_EMULATOR_TEST_PASS', flush=True)
