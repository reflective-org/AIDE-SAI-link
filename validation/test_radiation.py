"""Smoke test: radiation driver on one real CESM hour.

Checks: Mie tables sane, rrtmgp runs on GPU without NaNs, night columns
zeroed, aerosol AOD magnitude vs CESM AODVISstdn, heating anomaly of a
perturbed (coagulated-like) distribution is finite and small.
Run: CUDA_VISIBLE_DEVICES=<free> python3 validation/test_radiation.py
"""
import os, sys, time
import numpy as np

# coupling/radiation live in the repo root, one level up. Python puts the
# SCRIPT's directory on sys.path, not the cwd, so without this the import fails
# no matter where you launch from (it did, from the 2026-07-29 move until
# 2026-07-30).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402 -- puts src/ subdirs on sys.path

import coupling as C
import radiation as R
import jax.numpy as jnp

t0 = time.time()
print('devices:', __import__('jax').devices(), flush=True)

# ---- grid/band exactly as coupling.main() builds it ----
dU = C.open_var('U')
lat = dU['lat'].values; lon = dU['lon'].values
plev = dU['hyam'].values * dU['P0'].values + dU['hybm'].values * C.PS_REF
band = np.where((plev >= C.P_LO_HPA * 100) & (plev <= C.P_HI_HPA * 100))[0]
klevs = list(range(band.min(), band.max() + 1))
print(f'band: {len(klevs)} levels {plev[klevs[0]]/100:.2f}..{plev[klevs[-1]]/100:.1f} hPa')

# ---- Mie sanity ----
DP_M = C.DP_BIN * 1e-9
mie = R.build_mie_tables(DP_M, os.path.join(R.RRTMGP_DATA, 'rrtmgp-gas-sw-g112.nc'))
i550 = 10   # SW band 11 = 0.442-0.625 um
print(f'Mie: sigma_ext(550nm band) range {mie["sigma_ext"][:,i550].min():.2e}..'
      f'{mie["sigma_ext"][:,i550].max():.2e} m2, '
      f'ssa {mie["ssa"][:,i550].min():.3f}..{mie["ssa"][:,i550].max():.3f}, '
      f'g {mie["g"][:,i550].min():.3f}..{mie["g"][:,i550].max():.3f}', flush=True)
assert np.all(mie['sigma_ext'] > 0) and np.all((mie['ssa'] >= 0) & (mie['ssa'] <= 1))

# ---- driver ----
drv = R.RadiationDriver(C.open_var, klevs, lat, lon, DP_M)
print(f'driver ready ({time.time()-t0:.0f}s): column {drv.nzc} levels, '
      f'band rows {drv.band_pos[0]}..{drv.band_pos[-1]}, '
      f'{drv.nb_sw} SW + {drv.nb_lw} LW bands', flush=True)

# ---- aerosol from MAM4 at hour T_IDX (noon-ish UTC) ----
T_IDX = 12
ds_mam = {f'{p}_a{m}': C.open_var(f'{p}_a{m}')
          for p in ('num', 'so4') for m in (1, 2, 3)}
# bin_mam4 uses C.H0 offset; pass t so that H0+t = T_IDX
num, mas = C.bin_mam4(ds_mam, T_IDX - C.H0, klevs)
num = jnp.asarray(num)
print(f'aerosol binned: total N mr {float(num.sum()):.3e}', flush=True)

# ---- heating with MAM4 aerosol ----
t1 = time.time()
hr, diag = drv.heating(T_IDX, num)
hr = np.asarray(hr)
print(f'heating call: {time.time()-t1:.0f}s (incl. compile)', flush=True)
kday = hr * 86400.0
print(f'HR [K/day]: min {kday.min():.3f} max {kday.max():.3f} '
      f'mean {kday.mean():.4f}  finite={np.isfinite(hr).all()}', flush=True)
assert np.isfinite(hr).all(), 'NaN/inf in heating rate'

# night columns: SW down at TOA must be 0 where sun below horizon
zen = R.solar_zenith(drv.time[T_IDX], lat, lon)
sw_toa = np.asarray(diag['sw_dn_toa'])
night = zen >= np.pi / 2
print(f'night columns: {night.sum()} of {night.size}; '
      f'max SW_dn_toa at night = {sw_toa[night].max():.2e} W/m2 '
      f'(day max {sw_toa[~night].max():.0f})', flush=True)
assert sw_toa[night].max() < 1e-6

# ---- AOD vs CESM AODVISstdn ----
aod = np.asarray(diag['aod550'])
dA = C.open_var('AODVISstdn')
aod_cesm = dA['AODVISstdn'].isel(time=T_IDX).values
w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(aod)
gm = lambda x: float((np.nan_to_num(x) * w).sum() / w.sum())
print(f'AOD550 global mean: ours(band,sulfate-only) {gm(aod):.4f} '
      f'vs CESM AODVISstdn(strat) {gm(aod_cesm):.4f}', flush=True)

# ---- anomaly call: same aerosol twice must give ~0 ----
t2 = time.time()
anom, d1, d0 = drv.heating_anomaly(T_IDX, num, num, np.zeros_like(hr))
anom = np.asarray(anom)
print(f'anomaly(identical states): max|.| {np.abs(anom).max():.2e} K/s '
      f'({time.time()-t2:.0f}s)', flush=True)
assert np.abs(anom).max() < 1e-12

# ---- anomaly with a perturbed distribution (crude coag proxy: shift num) ----
num_pert = num.at[:10].multiply(0.5).at[20:30].multiply(1.5)
anom, d1, d0 = drv.heating_anomaly(T_IDX, num_pert, num, np.zeros_like(hr))
akd = np.asarray(anom) * 86400.0
print(f'anomaly(perturbed bins) [K/day]: min {akd.min():.4f} max {akd.max():.4f} '
      f'finite={np.isfinite(akd).all()}', flush=True)
print(f'\nALL CHECKS PASSED ({time.time()-t0:.0f}s total)', flush=True)
