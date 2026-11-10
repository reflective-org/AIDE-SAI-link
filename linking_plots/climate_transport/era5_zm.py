#!/usr/bin/env python3
"""Daily (06 UTC) zonal means of ERA5 u, v, T for 1997 on the climate_zm.npz band levels/lat grid.

Source: WeatherBench2 public ERA5, 37 pressure levels, hourly, 512x256 equiangular conservative
  gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-512x256_equiangular_conservative.zarr
(read chunk-by-chunk over HTTPS, only the stratospheric levels; nothing stored on disk).
The local JCM cache /data/jcm_era5_cache/wb2_192x96_l63_35c39d41_*.nc was NOT usable: it was built by
jcm/data/era5.py from the 13-level WB2 store (50-1000 hPa) and clamps every model level above 50 hPa
to the 50 hPa value.
ERA5 is already on pressure levels, so vertical interp (linear in log p) commutes with the zonal mean."""
import json, datetime as dt, numpy as np, requests, numcodecs
from concurrent.futures import ThreadPoolExecutor
B = 'https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-512x256_equiangular_conservative.zarr'
OUT = '/data/AIDE-SAI-link/linking_plots/climate_transport/data'
S = requests.Session(); S.mount('https://', requests.adapters.HTTPAdapter(pool_maxsize=64))

def get(path):
    for k in range(6):
        try:
            r = S.get(f'{B}/{path}', timeout=120); r.raise_for_status(); return r.content
        except Exception as e:
            err = e
    raise err

def arr1d(name):
    meta = json.loads(get(f'{name}/.zarray'))
    raw = numcodecs.Blosc().decode(get(f'{name}/0'))
    return np.frombuffer(raw, meta['dtype'])[:meta['shape'][0]]

lev = arr1d('level'); lat_s = arr1d('latitude'); tim = arr1d('time')
print('levels', lev); print('lat', lat_s[:3], lat_s[-1], 'time0', tim[0], tim[1])
d = np.load(f'{OUT}/climate_zm.npz'); lat = d['lat']; plev = d['plev']
dates = [dt.datetime(1997, 1, 1, 6) + dt.timedelta(days=i) for i in range(len(d['hours']))]
H = np.array([int((t - dt.datetime(1959, 1, 1)).total_seconds() // 3600) for t in dates])
TC = 70158                                              # time chunk length; check the relevant one is hourly-from-0
for c in sorted(set(H // TC)):
    tc = np.frombuffer(numcodecs.Blosc().decode(get(f'time/{c}')), '<i8')
    assert np.array_equal(tc[:len(tc)], np.arange(c * TC, c * TC + len(tc))), c
ti = H
kl = [i for i, l in enumerate(lev) if l <= 150]          # 1..150 hPa
ps = lev[kl].astype(float) * 100
print('using levels', lev[kl])
CH = 48
VARS = {'T': 'temperature', 'u': 'u_component_of_wind', 'v': 'v_component_of_wind'}

def job(args):
    var, c, k = args
    raw = numcodecs.Blosc().decode(get(f'{VARS[var]}/{c}.{k}.0.0'))
    a = np.frombuffer(raw, '<f4').reshape(CH, 512, 256)
    sel = [t - c * CH for t in ti if t // CH == c]
    return var, c, k, a[sel].mean(1)                     # zonal mean over lon -> (nsel, 256)

chunks = sorted(set(ti // CH))
tasks = [(v, c, k) for v in VARS for c in chunks for k in kl]
zm = {v: np.full((len(ti), len(kl), 256), np.nan, np.float32) for v in VARS}
pos = {t: i for i, t in enumerate(ti)}
with ThreadPoolExecutor(48) as ex:
    for n, (v, c, k, z) in enumerate(ex.map(job, tasks)):
        rows = [pos[t] for t in ti if t // CH == c]
        zm[v][rows, kl.index(k)] = z
        if n % 500 == 0: print(n, '/', len(tasks), flush=True)
assert all(np.isfinite(z).all() for z in zm.values())
np.savez(f'{OUT}/era5_zm_native.npz', lev_hpa=lev[kl], lat=lat_s, **{f'era5_{v}': z for v, z in zm.items()})
# lat -> 192 grid (linear, edge-clamped), then p -> band levels (linear in log p)
o = np.argsort(lat_s); lx = lat_s[o]; lp = np.log(ps); lt = np.log(plev)
res = dict(lat=lat, plev=plev, hours=d['hours'], idx=d['idx'], source=B)
for v, z in zm.items():
    z = z[..., o]
    zl = np.stack([[np.interp(lat, lx, z[t, k]) for k in range(len(kl))] for t in range(len(ti))])
    res[f'era5_{v}'] = np.stack([[np.interp(lt, lp, zl[t, :, j]) for j in range(len(lat))] for t in range(len(ti))]).transpose(0, 2, 1).astype(np.float32)
np.savez(f'{OUT}/era5_zm.npz', **res)
print('wrote', f'{OUT}/era5_zm.npz', {k: v.shape for k, v in res.items() if hasattr(v, 'shape')})
