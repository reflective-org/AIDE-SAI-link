#!/usr/bin/env python3
"""Cross-check daily-06Z ERA5 zonal means (WB2) against CDS monthly-mean zonal means for 1997."""
import numpy as np, xarray as xr, datetime as dt
OUT = '/data/AIDE-SAI-link/linking_plots/climate_transport/data'
n = np.load(f'{OUT}/era5_zm_native.npz')
ref = xr.open_dataset('/data/cesm-jax-dycore/references/era5_1995_1999.nc').sel(time=slice('1997-01', '1997-12'))
days = np.array([(dt.datetime(1997, 1, 1, 6) + dt.timedelta(days=i)).month for i in range(366)])[:-1]
lev = n['lev_hpa']; lat = n['lat']; o = np.argsort(lat)
lines = []
for v in ('T', 'u'):
    z = n[f'era5_{v}'][:-1][..., o]
    for P in (1, 2, 5, 10, 30, 50, 100):
        if P not in lev or P not in ref.plev.values: continue
        k = list(lev).index(P)
        mine = np.stack([[np.interp(ref.lat.values, lat[o], z[days == m, k].mean(0))] for m in range(1, 13)])[:, 0]
        r = ref[v].sel(plev=P).values
        dif = mine - r
        lines.append(f'{v} {P:>4} hPa: monthly RMS diff {np.sqrt((dif**2).mean()):.2f}, mean {dif.mean():+.2f}, max|d| {np.abs(dif).max():.2f}; '
                     f'ref range {r.min():.1f}..{r.max():.1f}')
# profiles
z = n['era5_T'][:-1][..., o]; L = lat[o]
eq = np.abs(L) <= 10
prof = z[:, :, eq].mean((0, 2))
lines.append('annual 10S-10N T profile: ' + ', '.join(f'{l}hPa {t:.1f}' for l, t in zip(lev, prof)))
lines.append('annual global(cos) T at 1 hPa: %.1f' % ((z[:, 0] * np.cos(np.deg2rad(L))).sum(-1) / np.cos(np.deg2rad(L)).sum()).mean())
open(f'{OUT}/era5_validation.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
