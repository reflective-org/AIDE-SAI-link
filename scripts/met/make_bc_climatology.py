#!/usr/bin/env python3
"""One-time pass over the CESM archive -> bc_clim.npz, the CESM-free boundary file.

docs/BOUNDARY_CONDITIONS.md: with the circulation coming from an emulator
(MET_SRC=file) the archive is still read for everything the emulator does not
carry. This file replaces ALL of those reads with a monthly, day-of-year-cyclic
climatology (coupling.py BC_SRC=clim serves it through the same open_var()
interface the archive has, so the model code does not change):

  3-D on the 70 native levels, (12, 70, 192, 288) float32 each:
    Q, O3, CH4         radiation column gases; Q also gives RH with the met T
    num_a1..3, so4_a1..3, dgnumwet1..3   MAM4 modes: aerosol IC, slabs, polar
                        caps, and the anomaly-mode reference aerosol
    SO2, H2SO4         gas IC and slabs
    OH, RELHUM, T      only read under OH_SZA=0 / MET_RH=cesm / diagnostics;
                        T is NOT the model's temperature under MET_SRC=file
  2-D (12, 192, 288): FLDS, FLNS, FSDS, FSNS (surface T and albedo), PS, TS
  grid: lat, lon, hyam, hybm, hyai, hybi, P0; months = calendar month centres

Sources. The monthly h0 stream (month_1) for everything it has: record i of
the 1996-01..2014-12 series IS month i (the stamp is the END of the period --
time 31 for January 1996 -- so records are taken by index, never by stamp).
H2SO4 has no monthly output, so it is a sub-sampled mean of the hourly h1
series: days 1, 8, 15, 22 of every month at 00Z and 12Z, all 19 years (1824
hours; the gas has a diurnal cycle and both phases are in the sample).

  python3 scripts/met/make_bc_climatology.py --out runs/bc_clim/bc_clim.npz [--years 1996-2014]

~2.6 GB on disk (float32, uncompressed; np.load(mmap_mode='r') reads it lazily).
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from emu_to_met import CESM_DIR, CESM_PREFIX, CESM_SUF  # noqa: E402

MONTHLY_3D = ('Q', 'O3', 'CH4', 'SO2', 'OH', 'RELHUM', 'T',
              'num_a1', 'num_a2', 'num_a3', 'so4_a1', 'so4_a2', 'so4_a3',
              'dgnumwet1', 'dgnumwet2', 'dgnumwet3')
MONTHLY_2D = ('FLDS', 'FLNS', 'FSDS', 'FSNS', 'PS', 'TS')
HOURLY_3D = ('H2SO4',)
MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def h0_file(var):
    fs = glob.glob(f'{CESM_DIR}/month_1/{CESM_PREFIX}.h0.{var}.*.nc')
    assert len(fs) == 1, (var, fs)
    return fs[0]


def monthly_clim(var, y0, y1, first_year=1996):
    """Mean over years of each calendar month, records taken by index."""
    ds = xr.open_dataset(h0_file(var), decode_times=False)
    da = ds[var]
    i0 = (y0 - first_year) * 12; i1 = (y1 - first_year + 1) * 12
    assert i1 <= da.sizes['time'], (var, da.sizes['time'], i1)
    # time_bnds[i] = [start, end] in days since 1996-01-01: check record 0 is Jan 1996
    tb = ds['time_bnds'].values
    assert tb[0, 0] == 0.0 and tb[0, 1] == 31.0, tb[0]
    out = np.zeros((12,) + da.shape[1:], np.float64)
    n = 0
    for y in range(y0, y1 + 1):
        j = (y - first_year) * 12
        out += da.isel(time=slice(j, j + 12)).values.astype(np.float64)
        n += 1
    ds.close()
    return (out / n).astype(np.float32), dict(units=str(da.attrs.get('units', '')),
                                                long_name=str(da.attrs.get('long_name', '')))


def hourly_clim(var, y0, y1):
    """Monthly climatology of an hourly variable from days 1/8/15/22 at 00Z and 12Z."""
    fn = f'{CESM_DIR}/hour_1/{CESM_PREFIX}.h1.{var}{CESM_SUF}'
    ds = xr.open_dataset(fn, decode_times=False)
    da = ds[var]
    out = np.zeros((12,) + da.shape[1:], np.float64)
    cnt = np.zeros(12)
    t0 = time.time()
    for y in range(y0, y1 + 1):
        day_of_year0 = (y - 1996) * 365
        for m in range(12):
            d0 = day_of_year0 + int(MONTH_DAYS[:m].sum())
            hours = [(d0 + d) * 24 + h for d in (0, 7, 14, 21) for h in (0, 12)]
            out[m] += da.isel(time=hours).values.astype(np.float64).sum(axis=0)
            cnt[m] += len(hours)
        print(f'    {var} {y} {time.time()-t0:.0f}s', flush=True)
    ds.close()
    return (out / cnt[:, None, None, None]).astype(np.float32), dict(
        units=str(da.attrs.get('units', '')), long_name=str(da.attrs.get('long_name', '')),
        sampling='days 1,8,15,22 at 00Z,12Z of every month')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--years', default='1996-2014')
    a = ap.parse_args()
    y0, y1 = (int(x) for x in a.years.split('-'))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ds = xr.open_dataset(h0_file('T'), decode_times=False)
    grid = dict(lat=ds['lat'].values.astype(np.float64), lon=ds['lon'].values.astype(np.float64),
                hyam=ds['hyam'].values.astype(np.float64), hybm=ds['hybm'].values.astype(np.float64),
                hyai=ds['hyai'].values.astype(np.float64), hybi=ds['hybi'].values.astype(np.float64),
                P0=np.float64(ds['P0'].values), lev=ds['lev'].values.astype(np.float64))
    ds.close()
    fields, attrs = {}, {}
    t0 = time.time()
    for var in MONTHLY_3D + MONTHLY_2D:
        fields[var], attrs[var] = monthly_clim(var, y0, y1)
        print(f'  {var} {fields[var].shape} {time.time()-t0:.0f}s', flush=True)
    for var in HOURLY_3D:
        fields[var], attrs[var] = hourly_clim(var, y0, y1)
        print(f'  {var} {fields[var].shape} {time.time()-t0:.0f}s', flush=True)
    # month centres in day-of-year (noleap), for the runtime interpolation
    month_centre_doy = np.cumsum(MONTH_DAYS) - MONTH_DAYS / 2.0
    np.savez(a.out, **fields, **grid, month_centre_doy=month_centre_doy,
             fields_3d=np.array(MONTHLY_3D + HOURLY_3D), fields_2d=np.array(MONTHLY_2D),
             attrs=np.array([f'{k}: {v}' for k, v in attrs.items()]),
             provenance=np.array(f'{CESM_PREFIX} month_1 (by record index) + h1 sub-sample for H2SO4; '
                                 f'years {y0}-{y1}; built {time.strftime("%Y-%m-%d")} by make_bc_climatology.py'),
             years=np.array([y0, y1]), calendar=np.array('noleap'))
    print(f'wrote {a.out} ({os.path.getsize(a.out)/1e9:.2f} GB) in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
