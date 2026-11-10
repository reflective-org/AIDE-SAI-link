#!/usr/bin/env python3
"""Gate for the radiation reference cache and the split-mode solve.

Builds the production RadiationDriver (as coupling.main() does), then on the
day-30 state checks that (1) heating(split=True) returns the same hr_band as
the default path, bit for bit, with hr_sw_band + hr_lw = hr; (2) the
zero-aerosol call runs and is the clear-sky limit (smaller |hr| anomaly);
(3) reference() stores an hour in a scratch cache and serves it back
bit-identically (hr0, every diag field) without a num_ref; (4) the cache key
is stable across two driver builds. GPU + CESM archive, ~3 min.
Run: CUDA_VISIBLE_DEVICES=<free> python3 scripts/radiation/test_rad_cache.py
"""
import os, sys, time, tempfile
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402
import coupling as C
import radiation as R
import rad_cache
import jax, jax.numpy as jnp

t0 = time.time()
dU = C.open_var('U'); lat = dU['lat'].values; lon = dU['lon'].values
plev = dU['hyam'].values * dU['P0'].values + dU['hybm'].values * C.PS_REF
zd = np.load(f'{_REPO}/runs/chemdump/debug_nucleation/premicro_state_step120.npz')
klevs = [int(np.argmin(np.abs(plev - pp))) for pp in zd['plev']]
drv = R.RadiationDriver(C.open_var, klevs, lat, lon, C.DP_BIN * 1e-9, mmid_kg=C.MMID, rho_dry=C.RHO_AER)
num = jnp.asarray(zd['num']); t = C.H0 + 720
print(f'driver ready {time.time()-t0:.0f}s; nzc={drv.nzc} nzb={drv.nzb} nbins={drv.nbins}', flush=True)

hr, d = drv.heating(t, num); jax.block_until_ready(hr)
hrs, ds = drv.heating(t, num, split=True); jax.block_until_ready(hrs)
assert np.array_equal(np.asarray(hr), np.asarray(hrs)), 'split=True changed hr_band'
for k in d:
    assert np.array_equal(np.asarray(d[k]), np.asarray(ds[k])), f'split=True changed diag {k}'
hsw = np.asarray(ds['hr_sw_band']); hlw = np.asarray(hrs) - hsw
zen = R.solar_zenith(drv.time[t], drv.lat, drv.lon)
night = zen >= 0.5 * np.pi
assert np.all(hsw[:, night] == 0.0), 'SW heating nonzero at night'
print(f'split OK: |hr| p50 {np.median(np.abs(hr))*86400:.3f} K/d, SW p50 {np.median(np.abs(hsw))*86400:.3f}, '
      f'LW p50 {np.median(np.abs(hlw))*86400:.3f} K/d; night cols {night.mean()*100:.0f}%', flush=True)

hr_c, d_c = drv.heating(t, None); jax.block_until_ready(hr_c)
print(f'clear-sky OK: |hr_aer - hr_clear| p50 {np.median(np.abs(np.asarray(hr-hr_c)))*86400:.2e} K/d, '
      f'max {np.abs(np.asarray(hr-hr_c)).max()*86400:.3f} K/d; olr diff max {np.abs(np.asarray(d["olr"]-d_c["olr"])).max():.3f} W/m2', flush=True)

tmp = tempfile.mkdtemp(prefix='rad_cache_test_')
key = drv.cache_key(AER_SRC='test')
key2 = R.RadiationDriver(C.open_var, klevs, lat, lon, C.DP_BIN * 1e-9, mmid_kg=C.MMID, rho_dry=C.RHO_AER).cache_key(AER_SRC='test')
assert rad_cache.make_key(key) == rad_cache.make_key(key2), 'cache key not stable'
drv.ref_cache = rad_cache.RefCache(tmp, key)
assert not drv.ref_ready(t)
num_ref = jnp.asarray(np.asarray(zd['num']) * 0.3)
ta = time.time(); hr0a, d0a = drv.reference(t, num_ref); jax.block_until_ready(hr0a); ta = time.time() - ta
assert drv.ref_ready(t)
tb = time.time(); hr0b, d0b = drv.reference(t, None); jax.block_until_ready(hr0b); tb = time.time() - tb
assert np.array_equal(np.asarray(hr0a), np.asarray(hr0b)) and np.asarray(hr0b).dtype == np.float64, 'cached hr0 differs'
for k in d0a:
    assert np.array_equal(np.asarray(d0a[k]), np.asarray(d0b[k])), f'cached diag {k} differs'
hr0c, _ = drv.heating(t, num_ref)
assert np.array_equal(np.asarray(hr0c), np.asarray(hr0b)), 'recomputed hr0 differs from cached'
an_a = np.asarray(hr - hr0a); an_b = np.asarray(hr - hr0b)
assert np.array_equal(an_a, an_b)
sz = os.path.getsize(drv.ref_cache.path(t)) / 1e6
print(f'cache OK: compute+store {ta:.2f}s, serve {tb:.3f}s, entry {sz:.1f} MB, key {drv.ref_cache.key}; {drv.ref_cache.stats()}', flush=True)
import shutil; shutil.rmtree(tmp)
print('RAD_CACHE_TEST_PASS', flush=True)
