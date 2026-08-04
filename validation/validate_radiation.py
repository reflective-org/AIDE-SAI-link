"""Validate the radiation driver against CESM's own radiation output.

1. Heating rates: our clear-sky rrtmgp heating (with MAM4-binned sulfate)
   vs CESM monthly QRS+QRL (all-sky, full aerosol) for Jan 1996, zonal mean.
   Expect close agreement in the stratosphere (clear sky there), and
   cloud-driven differences in the troposphere.
2. Aerosol extinction: our per-layer 550nm extinction from the binned MAM4
   sulfate vs CESM's hourly EXTINCTdn profile, zonal mean at one hour.
   Validates the bin->Mie->optics chain independently of the RTE solve.

Run: CUDA_VISIBLE_DEVICES=<free> python3 validation/validate_radiation.py
"""
import os, sys, glob, time
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# coupling/radiation live in the repo root, one level up -- see the note in
# test_radiation.py. sys.path[0] is this file's directory, not the cwd.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'fast_advection'))

import coupling as C
import radiation as R
import jax.numpy as jnp

t0 = time.time()
MDIR = os.path.join(C.HDIR, 'month_1')

def open_month(var):
    fn = glob.glob(f'{MDIR}/{C.PREFIX}.h0.{var}.*.nc')[0]
    return xr.open_dataset(fn)

# ---- grid/band as in coupling.main() ----
dU = C.open_var('U')
lat = dU['lat'].values; lon = dU['lon'].values
plev = dU['hyam'].values * dU['P0'].values + dU['hybm'].values * C.PS_REF
band = np.where((plev >= C.P_LO_HPA * 100) & (plev <= C.P_HI_HPA * 100))[0]
klevs = list(range(band.min(), band.max() + 1))
PLEV_B = plev[klevs] / 100.0                      # hPa, band levels

DP_M = C.DP_BIN * 1e-9
drv = R.RadiationDriver(C.open_var, klevs, lat, lon, DP_M)
ds_mam = {f'{p}_a{m}': C.open_var(f'{p}_a{m}')
          for p in ('num', 'so4') for m in (1, 2, 3)}

# ---- 1. heating: average our HR over 4 times of day (Jan 2 1996) ----
HOURS = [24, 30, 36, 42]                          # 00/06/12/18 UTC on Jan 2
hr_sum = None
for t in HOURS:
    num, _ = C.bin_mam4(ds_mam, t - C.H0, klevs)
    hr, _ = drv.heating(t, jnp.asarray(num))
    hr_sum = np.asarray(hr) if hr_sum is None else hr_sum + np.asarray(hr)
    print(f'  hour {t} done ({time.time()-t0:.0f}s)', flush=True)
hr_ours = hr_sum / len(HOURS) * 86400.0           # K/day, (nzb, nlat, nlon)

dQRS = open_month('QRS'); dQRL = open_month('QRL')
qrs = dQRS['QRS'].isel(time=0, lev=klevs).values  # K/s, Jan 1996 monthly mean
qrl = dQRL['QRL'].isel(time=0, lev=klevs).values
hr_cesm = (qrs + qrl) * 86400.0                   # K/day

zon_ours = hr_ours.mean(axis=2)                   # (nzb, nlat)
zon_cesm = hr_cesm.mean(axis=2)

# ---- 2. extinction at one hour (Jan 2, 12 UTC) ----
T_EXT = 36
num, _ = C.bin_mam4(ds_mam, T_EXT - C.H0, klevs)
sig550, _ = R.mie_at_wavelength(DP_M, 0.550)
dT_ds = C.open_var('T')
Tf = dT_ds['T'].isel(time=T_EXT, lev=klevs).values
rho = np.broadcast_to(plev[klevs][:, None, None], Tf.shape) / (C.RD * Tf)
# num [#/kg] * rho [kg/m3] = #/m3; * sigma [m2] = 1/m; * 1e3 = 1/km
ext_ours = np.einsum('k,kzyx->zyx', sig550, num * rho[None]) * 1e3
dE = C.open_var('EXTINCTdn')
ext_cesm = dE['EXTINCTdn'].isel(time=T_EXT, lev=klevs).values * 1e3  # /m -> /km
ez_ours = ext_ours.mean(axis=2); ez_cesm = np.nan_to_num(ext_cesm).mean(axis=2)

# ---- plots ----
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
vmax = np.nanpercentile(np.abs(zon_cesm), 99)
for i, (z, t) in enumerate([(zon_ours, 'ours (clear-sky, MAM4 sulfate)'),
                            (zon_cesm, 'CESM QRS+QRL (all-sky, monthly)')]):
    im = ax[0, i].pcolormesh(lat, PLEV_B, z, cmap='RdBu_r',
                             vmin=-vmax, vmax=vmax, shading='auto')
    ax[0, i].invert_yaxis(); ax[0, i].set_yscale('log')
    ax[0, i].set_title(f'Net heating [K/day]: {t}')
    ax[0, i].set_xlabel('lat'); ax[0, i].set_ylabel('p [hPa]')
    plt.colorbar(im, ax=ax[0, i])

# profile comparison at a few latitudes
for j, lt in enumerate([-45, 0, 45]):
    i = np.argmin(np.abs(lat - lt))
    ax[0, 2].plot(zon_ours[:, i], PLEV_B, f'C{j}-', label=f'ours {lt}N')
    ax[0, 2].plot(zon_cesm[:, i], PLEV_B, f'C{j}--', label=f'CESM {lt}N')
ax[0, 2].invert_yaxis(); ax[0, 2].set_yscale('log')
ax[0, 2].set_xlabel('K/day'); ax[0, 2].set_ylabel('p [hPa]')
ax[0, 2].set_title('Heating profiles'); ax[0, 2].legend(fontsize=7)
ax[0, 2].grid(alpha=0.3)

emax = max(np.nanpercentile(ez_cesm, 99.5), 1e-6)
for i, (e, t) in enumerate([(ez_ours, 'ours (sulfate only)'),
                            (ez_cesm, 'CESM EXTINCTdn (all aerosol)')]):
    im = ax[1, i].pcolormesh(lat, PLEV_B, np.maximum(e, 0), cmap='viridis',
                             vmin=0, vmax=emax, shading='auto')
    ax[1, i].invert_yaxis(); ax[1, i].set_yscale('log')
    ax[1, i].set_title(f'550nm extinction [1/km]: {t}')
    ax[1, i].set_xlabel('lat'); ax[1, i].set_ylabel('p [hPa]')
    plt.colorbar(im, ax=ax[1, i])

# stratosphere-only profile (global mean)
istr = PLEV_B < 150
w = np.cos(np.deg2rad(lat))
gm = lambda z: (z * w[None, :]).sum(1) / w.sum()
ax[1, 2].plot(gm(ez_ours), PLEV_B, 'C0-', label='ours')
ax[1, 2].plot(gm(ez_cesm), PLEV_B, 'C1--', label='CESM')
ax[1, 2].invert_yaxis(); ax[1, 2].set_yscale('log'); ax[1, 2].set_ylim(1000, 1)
ax[1, 2].set_xlabel('extinction [1/km]'); ax[1, 2].set_ylabel('p [hPa]')
ax[1, 2].set_xscale('log'); ax[1, 2].set_title('Global-mean extinction profile')
ax[1, 2].legend(); ax[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('radiation_validation.png', dpi=130)
print(f'wrote radiation_validation.png ({time.time()-t0:.0f}s)')

# summary stats: stratospheric band (150-5 hPa) zonal-mean agreement
sb = (PLEV_B < 150) & (PLEV_B > 5)
d = zon_ours[sb] - zon_cesm[sb]
print(f'strat heating (5-150 hPa): ours {zon_ours[sb].mean():+.3f} '
      f'CESM {zon_cesm[sb].mean():+.3f} K/day, '
      f'bias {d.mean():+.3f}, rms diff {np.sqrt((d**2).mean()):.3f}')
sbe = ez_cesm[sb] > 1e-8
r = ez_ours[sb][sbe] / ez_cesm[sb][sbe]
print(f'strat extinction ratio ours/CESM: median {np.median(r):.3f} '
      f'(sulfate-only vs all-aerosol, so <=1 expected)')
