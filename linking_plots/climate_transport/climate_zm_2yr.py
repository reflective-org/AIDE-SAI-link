#!/usr/bin/env python3
"""Daily zonal-mean T and u, 1997-01-01 .. 1998-12-31 (730 days), on 40 log-spaced pressure levels 1-300 hPa,
for the DJF/JJA two-year climate comparison (2026-11-09).

Sources (each on its own latitude grid; vertical = per-column log-p interpolation from the source's own hybrid levels
with its own surface pressure, then the zonal mean):
  emu    P2@200 emulator, T63L95 frames, 06 UTC  (runs/emu_winds_P2s200_1997/frames_1997-01-01 + ..._1998/frames_1998-01-01)
  hlc    'hlcorpus' JCM dycore corpus (ERA5 nudge p>=417 hPa, HL drag hlr_p, no QBO nudge) = P2's training data, 06 UTC
  c1     'jcm_p20bt_c1' JCM dycore corpus (ERA5 + QBO nudged) = the base model's training data, 06 UTC
  cesm   CESM2 FWHIST fixedSST 001, DAILY MEANS (h7 day_1 T, U, PS); day selected by the lower time bound
ERA5 is done separately (era5_zm_2yr.py, network).
usage: climate_zm_2yr.py SRC [NW]     -> data/czm2_<SRC>.npz : T, u (730, 40, nlat), lat, p_hpa, dates"""
import sys, os, json, datetime as dt
import numpy as np
from multiprocessing import Pool
SRC = sys.argv[1]; NW = int(sys.argv[2]) if len(sys.argv) > 2 else 6
OUT = f'/data/AIDE-SAI-link/linking_plots/climate_transport/data/czm2_{SRC}' + ('_TEST' if os.environ.get('TEST') else '') + '.npz'
P_T = np.exp(np.linspace(np.log(1.0), np.log(300.0), 40)) * 100.0          # Pa, top->bottom (ascending p)
ND = 730
D0 = dt.date(1997, 1, 1)
CO = np.load('/data/paradis_jcm/jcm_analytic_p20bt.npz')
HYAM, HYBM = CO['hyam_pa'].astype(np.float64), CO['hybm'].astype(np.float64)   # surface-first, 95


def zm_on_p(F, P):
    """F, P: (nz, ny, nx) with P ascending along axis 0. Linear-in-log-p onto P_T, then zonal mean -> (40, ny).
    Targets outside a column's range are NaN (not extrapolated)."""
    nz = F.shape[0]; lp = np.log(P); out = np.empty((P_T.size,) + F.shape[1:])
    for j, pt in enumerate(np.log(P_T)):
        k = (lp < pt).sum(0) - 1                                  # index of the level just above pt
        ok = (k >= 0) & (k < nz - 1); kc = np.clip(k, 0, nz - 2)
        p0 = np.take_along_axis(lp, kc[None], 0)[0]; p1 = np.take_along_axis(lp, kc[None] + 1, 0)[0]
        f0 = np.take_along_axis(F, kc[None], 0)[0]; f1 = np.take_along_axis(F, kc[None] + 1, 0)[0]
        wgt = (pt - p0) / (p1 - p0)
        out[j] = np.where(ok, f0 + wgt * (f1 - f0), np.nan)
    return np.nanmean(out, -1) if np.isfinite(out).any() else out.mean(-1)


def jcm_day(args):
    """JCM-grid state (95 surface-first levels): returns zm T, u for one day."""
    i, getter = args
    u, T, ps = getter(i)
    P = (HYAM[:, None, None] + HYBM[:, None, None] * ps[None])[::-1]      # ascending p
    return i, zm_on_p(T[::-1].astype(np.float64), P), zm_on_p(u[::-1].astype(np.float64), P)


if SRC == 'emu':
    FR = {1997: '/data/AIDE-SAI-link/runs/emu_winds_P2s200_1997/frames_1997-01-01',
          1998: '/data/AIDE-SAI-link/runs/emu_winds_P2s200_1998/frames_1998-01-01'}
    lat = np.load('/data/paradis_jcm/jcm_analytic_p20bt.npz')['lat']
    def get(i):
        y = 1997 + i // 365; d = i % 365; f = FR[y]
        a = {k: np.load(f'{f}/{k}.npy', mmap_mode='r') for k in ('u', 'T', 'ps_norm')}
        return np.asarray(a['u'][4 * d]), np.asarray(a['T'][4 * d]), np.asarray(a['ps_norm'][4 * d], np.float64) * 1e5
elif SRC in ('hlc', 'c1'):
    import xarray as xr
    ROOT = {'hlc': '/raid/jcm_gen_states/hlcorpus', 'c1': '/mnt/nvme_stage2/jcm_p20bt_c1'}[SRC]
    ds0 = xr.open_zarr(f'{ROOT}/1997', consolidated=True); feats = [str(s) for s in ds0.features.values]
    lat = ds0.latitude.values
    iu = [feats.index(f'u_wind_h{k}') for k in range(95)]
    iT = [feats.index(next(s for s in feats if s.endswith(f'_h{k}') and s.startswith('t'))) for k in range(95)]
    ips = feats.index('normalized_surface_pressure')
    print('T channels', feats[iT[0]], feats[iT[-1]], 'ps', feats[ips], flush=True)
    def get(i):
        y = 1997 + i // 365; d = i % 365
        ds = xr.open_zarr(f'{ROOT}/{y}', consolidated=True)
        assert str(ds.time.values[4 * d])[:13] == f'{D0 + dt.timedelta(days=i)}T06', (ds.time.values[4 * d], i)
        x = ds.data.isel(time=4 * d).values                           # (96, 192, 381)
        return np.moveaxis(x[..., iu], -1, 0), np.moveaxis(x[..., iT], -1, 0), x[..., ips].astype(np.float64) * 1e5
elif SRC == 'cesm':
    import xarray as xr
    DD = '/data/cesm2.1.5_output/histSST/f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001/archive/atm/proc/tseries/day_1'
    pre = 'f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001.cam.h7'
    lat = xr.open_dataset(f'{DD}/{pre}.PS.19960101-20141231.nc').lat.values


def cesm_day(i):
    import xarray as xr
    ds = {v: xr.open_dataset(f'{DD}/{pre}.{v}.19960101-20141231.nc', decode_times=True) for v in ('T', 'U', 'PS')}
    tb = ds['PS'].time_bnds.values[:, 0] if 'time_bnds' in ds['PS'] else ds['PS'].time.values
    want = D0 + dt.timedelta(days=i)
    k = [n for n, t in enumerate(tb) if (t.year, t.month, t.day) == (want.year, want.month, want.day)]
    assert len(k) == 1, (i, k); k = k[0]
    h = ds['T']; P0 = float(h.P0) if 'P0' in h else 1e5
    ps = ds['PS'].PS.isel(time=k).values.astype(np.float64)
    P = h.hyam.values[:, None, None] * P0 + h.hybm.values[:, None, None] * ps[None]      # top-first = ascending
    T = h.T.isel(time=k).values.astype(np.float64); u = ds['U'].U.isel(time=k).values.astype(np.float64)
    for v in ds.values(): v.close()
    return i, zm_on_p(T, P), zm_on_p(u, P)


def day_fn(i):
    return cesm_day(i) if SRC == 'cesm' else jcm_day((i, get))


if __name__ == '__main__':
    os.nice(15)
    res_T = np.full((ND, P_T.size, lat.size), np.nan, np.float32); res_u = res_T.copy()
    days = range(ND) if not os.environ.get('TEST') else [0, 59, 400]
    with Pool(NW) as pool:
        for n, (i, zt, zu) in enumerate(pool.imap_unordered(day_fn, days)):
            res_T[i], res_u[i] = zt, zu
            if n % 50 == 0: print(SRC, n, '/', ND, flush=True)
    dates = np.array([str(D0 + dt.timedelta(days=i)) for i in range(ND)])
    np.savez(OUT, T=res_T, u=res_u, lat=lat, p_hpa=P_T / 100, dates=dates,
             note=f'{SRC}: daily zonal mean, log-p interpolated per column, 1-300 hPa; cesm = daily means, others 06 UTC')
    print('wrote', OUT, 'NaN frac T', float(np.isnan(res_T).mean()), 'u', float(np.isnan(res_u).mean()))
