#!/usr/bin/env python3
"""Age-of-air references (2026-11-09), run with /data/JCM_stripped/jcm-strat/.venv/bin/python (has h5netcdf).
  CLaMS-ERA5 (the 'ERA5 age of air'): /data/CLaMS/CLaMS_v3/clams_reanalyses/clams_era5_zm_lat_press.zip, BA (mean age
    clock tracer, yr; surface clock), 2000-2014 monthly zonal means; climatology over all 15 years (annual, DJF, JJA).
    off150 = its 0-10 deg (both hemispheres, cos-weighted) age at the level nearest 150 hPa: the jcm-strat sweep_score
    convention for comparing a 150-hPa-reset 'entry age' (CLaMS - off150).
  WACCM6 REF-D1.04 AOA (the 'CESM age of air'): /data/CESM2_REFD1_AOA/AoA_waccm6_refd1.04_AOA1mf_1970-2019_ba_0_100.0_ck_0_50.0.nc,
    monthly, 'date' stamps the END of the month (19970201 = Jan 1997), so 1997-1998 = 19970201..19990101 (24 months).
    Entry age based at 0.47N, 103 hPa; values < 0 (below the base) masked as in aoa_vs_clams.py.
Output data/aoa_refs.npz."""
import zipfile, io, numpy as np, xarray as xr
OUT = '/data/AIDE-SAI-link/linking_plots/climate_transport/data/aoa_refs.npz'
z = zipfile.ZipFile('/data/CLaMS/CLaMS_v3/clams_reanalyses/clams_era5_zm_lat_press.zip')
A = []
for n in sorted(z.namelist()):
    ds = xr.open_dataset(io.BytesIO(z.read(n)), decode_times=False)
    a = ds.BA.values.astype(np.float64); a[a < -1e20] = np.nan; A.append(a); p_c = ds.press.values; lat_c = ds.lat.values
A = np.stack(A)                                                     # (15, 12, np, nlat)
mon = np.arange(1, 13)
clams = {'ann': np.nanmean(A, (0, 1)), 'DJF': np.nanmean(A[:, np.isin(mon, (12, 1, 2))], (0, 1)), 'JJA': np.nanmean(A[:, np.isin(mon, (6, 7, 8))], (0, 1))}
k150 = int(np.argmin(abs(p_c - 150))); tr = np.abs(lat_c) <= 10; w = np.cos(np.deg2rad(lat_c))[tr]
off150 = float((clams['ann'][k150, tr] * w).sum() / w.sum())
print('CLaMS years', len(A), 'levels', p_c.size, 'off150 at', p_c[k150], 'hPa =', round(off150, 3), 'yr')
wd = xr.open_dataset('/data/CESM2_REFD1_AOA/AoA_waccm6_refd1.04_AOA1mf_1970-2019_ba_0_100.0_ck_0_50.0.nc', decode_times=False)
d = wd.date.values; sel = (d > 19970101) & (d <= 19990101); assert sel.sum() == 24
W = wd.AOA.values[sel].astype(np.float64); W[W < 0] = np.nan
mw = np.array([int(str(x)[4:6]) - 1 or 12 for x in d[sel]])          # month of the mean (stamp is the end)
waccm = {'ann': np.nanmean(W, 0), 'DJF': np.nanmean(W[np.isin(mw, (12, 1, 2))], 0), 'JJA': np.nanmean(W[np.isin(mw, (6, 7, 8))], 0)}
print('WACCM months', sorted(set(mw)), 'lev', wd.lev.values[[0, -1]])
np.savez(OUT, clams_p=p_c, clams_lat=lat_c, waccm_p=wd.lev.values, waccm_lat=wd.lat.values, clams_off150=off150, clams_p150=p_c[k150],
         **{f'clams_{k}': v for k, v in clams.items()}, **{f'waccm_{k}': v for k, v in waccm.items()})
print('wrote', OUT)
