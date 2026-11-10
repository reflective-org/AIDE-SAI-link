#!/usr/bin/env python3
"""Daily (06 UTC) zonal-mean ERA5 T and u, 1997-01-01 .. 1998-12-31, pressure levels 1..300 hPa, native 256 lat.
Same source and chunk reading as era5_zm.py (WeatherBench2 37-level hourly 512x256 conservative, streamed over HTTPS,
nothing stored but the zonal means). Output data/czm2_era5.npz: T, u (730, nlev, 256), lat (ascending), p_hpa (ERA5 levels).
Concurrency kept at 12 streams so /data NFS traffic from the coupled runs is not starved."""
import json, datetime as dt, numpy as np, requests, numcodecs, os
from concurrent.futures import ThreadPoolExecutor
B = 'https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-512x256_equiangular_conservative.zarr'
OUT = '/data/AIDE-SAI-link/linking_plots/climate_transport/data/czm2_era5.npz'
S = requests.Session(); S.mount('https://', requests.adapters.HTTPAdapter(pool_maxsize=32))

def get(path):
    err = None
    for k in range(8):
        try:
            r = S.get(f'{B}/{path}', timeout=180); r.raise_for_status(); return r.content
        except Exception as e:
            err = e
    raise err

def arr1d(name):
    meta = json.loads(get(f'{name}/.zarray'))
    return np.frombuffer(numcodecs.Blosc().decode(get(f'{name}/0')), meta['dtype'])[:meta['shape'][0]]

lev = arr1d('level'); lat_s = arr1d('latitude')
ND = 730
dates = [dt.datetime(1997, 1, 1, 6) + dt.timedelta(days=i) for i in range(ND)]
H = np.array([int((t - dt.datetime(1959, 1, 1)).total_seconds() // 3600) for t in dates])
TC = 70158
for c in sorted(set(H // TC)):                         # time axis must be hourly from 1959-01-01T00 (checked per chunk)
    tc = np.frombuffer(numcodecs.Blosc().decode(get(f'time/{c}')), '<i8')
    assert np.array_equal(tc, np.arange(c * TC, c * TC + len(tc))), c
kl = [i for i, l in enumerate(lev) if l <= 300]
print('levels', lev[kl], flush=True)
CH = 48
VARS = {'T': 'temperature', 'u': 'u_component_of_wind'}

def job(args):
    var, c, k = args
    a = np.frombuffer(numcodecs.Blosc().decode(get(f'{VARS[var]}/{c}.{k}.0.0')), '<f4').reshape(CH, 512, 256)
    sel = [t - c * CH for t in H if t // CH == c]
    return var, c, k, a[sel].mean(1)

chunks = sorted(set(H // CH)); pos = {t: i for i, t in enumerate(H)}
tasks = [(v, c, k) for v in VARS for c in chunks for k in kl]
zm = {v: np.full((ND, len(kl), 256), np.nan, np.float32) for v in VARS}
with ThreadPoolExecutor(int(os.environ.get('NTHREAD', '12'))) as ex:
    for n, (v, c, k, z) in enumerate(ex.map(job, tasks)):
        rows = [pos[t] for t in H if t // CH == c]
        zm[v][rows, kl.index(k)] = z
        if n % 500 == 0: print(n, '/', len(tasks), flush=True)
assert all(np.isfinite(z).all() for z in zm.values())
o = np.argsort(lat_s)
np.savez(OUT, T=zm['T'][..., o], u=zm['u'][..., o], lat=lat_s[o], p_hpa=lev[kl].astype(float),
         dates=np.array([str(d.date()) for d in dates]), note='ERA5 (WB2 37-level 512x256), 06 UTC, zonal mean, native levels')
print('wrote', OUT)
