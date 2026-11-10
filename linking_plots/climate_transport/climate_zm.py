#!/usr/bin/env python3
"""Daily zonal means (every 4th 6-h frame) of the emulator met (u, v, omega, T)
and the CESM archive at the same hours/levels (u, T)."""
import json, sys, numpy as np, xarray as xr
sys.path.insert(0, '/data/AIDE-SAI-link/scripts/met')
from emu_to_met import CESM_DIR, CESM_PREFIX, CESM_SUF
MET = '/data/AIDE-SAI-link/runs/emu_winds_P2s200_1997/met'
OUT = '/data/AIDE-SAI-link/linking_plots/climate_transport/data/climate_zm.npz'
meta = json.load(open(f'{MET}/meta.json'))
idx = np.arange(0, 1461, 4)              # 1997-01-01T06 .. 1998-01-01T06, daily
hours = meta['hour0'] + 6 * idx
res = dict(lat=np.array(meta['lat']), plev=np.array(meta['plev_pa']), idx=idx, hours=hours)
for k in ('u', 'v', 'omega', 'T'):
    a = np.load(f'{MET}/{k}.npy', mmap_mode='r')
    res[f'emu_{k}'] = np.stack([np.asarray(a[i], np.float64).mean(-1) for i in idx]).astype(np.float32)
    print('emu', k, flush=True)
for var, k in (('U', 'u'), ('T', 'T'), ('V', 'v'), ('OMEGA', 'omega')):
    ds = xr.open_dataset(f'{CESM_DIR}/hour_1/{CESM_PREFIX}.h1.{var}{CESM_SUF}')
    res[f'cesm_{k}'] = ds[var].isel(time=hours, lev=meta['klevs']).mean('lon').values.astype(np.float32)
    ds.close(); print('cesm', k, flush=True)
np.savez(OUT, **res)
print('wrote', OUT)
