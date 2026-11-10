#!/usr/bin/env python3
"""CESM h1 U/V/OMEGA at the emulator met file's hours and band levels (6-hourly, 1461 frames),
same layout as runs/emu_winds_P2s200_1997/met, for the CESM-wind AOA control."""
import json, sys, numpy as np, xarray as xr
sys.path.insert(0, '/data/AIDE-SAI-link/scripts/met')
from emu_to_met import CESM_DIR, CESM_PREFIX, CESM_SUF
MET = '/data/AIDE-SAI-link/runs/emu_winds_P2s200_1997/met'
OUT = '/data/AIDE-SAI-link/linking_plots/climate_transport/data/met_cesm1997'
meta = json.load(open(f'{MET}/meta.json'))
hours = meta['hour0'] + 6 * np.arange(1461)
for var, k in (('U', 'u'), ('V', 'v'), ('OMEGA', 'omega')):
    ds = xr.open_dataset(f'{CESM_DIR}/hour_1/{CESM_PREFIX}.h1.{var}{CESM_SUF}')
    assert np.allclose(ds['lat'].values, meta['lat'])
    out = np.lib.format.open_memmap(f'{OUT}/{k}.npy', 'w+', np.float32, (1461, 24, 192, 288))
    for a in range(0, 1461, 61):
        out[a:a + 61] = ds[var].isel(time=hours[a:a + 61], lev=meta['klevs']).values.astype(np.float32)
        print(var, a, flush=True)
    out.flush(); del out; ds.close()
m = dict(meta); m['source'] = 'cesm-archive'; m['nframe'] = 1461
json.dump(m, open(f'{OUT}/meta.json', 'w'))
print('wrote', OUT)
