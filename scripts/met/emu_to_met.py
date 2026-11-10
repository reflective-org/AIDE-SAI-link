#!/usr/bin/env python3
"""Regrid JCM-emulator free-run frames onto the coupled band -> a MET_SRC=file directory.

WHAT THIS IS
------------
The coupled model (src/coupling.py, MET_SRC=file) reads u, v, omega, T at the
START and END of every 6-h step on ITS grid: the CESM f09 lat/lon (192 x 288,
poles included) and the coupled band levels (klevs, pure-pressure levels of the
CESM hybrid grid at PS_REF = 1e5 Pa). The JCM emulator (P2@200 and kin) writes
its free run on the T63 Gaussian grid (96 x 192, no pole rows) and 95 hybrid
levels, surface first. This script maps one onto the other once, offline, so
the coupled run never carries torch or the emulator grid:

  vertical  : linear in log p per column, from the emulator's p_k = a_k + b_k ps
              (b_k <= 0.0074 over the band, so ps matters by < 0.4 hPa at the
              band bottom; done exactly anyway) onto the band's fixed pressures.
              The band spans 1.24-143 hPa; emulator levels 26..74 bracket it.
  horizontal: bilinear in (lat, lon) from the Gaussian grid to f09. The two
              f09 pole rows lie outside the Gaussian range (|lat| > 88.57) and
              take the nearest Gaussian row (constant extrapolation); the
              coupled transport stirs |lat| > 80 into one cell per level
              anyway (ADV_POLAR=zonal), so nothing is read from those rows
              except through that mixing. Longitude is periodic.
  time      : frame i is t0 + 6 h * i. meta.json records the absolute hour
              index of frame 0 on the CESM h1 axis (hours since 1996-01-01 00Z
              of the 1996-2014 archive, noleap), so coupling.py can map its
              hour counter H0 + t onto frames and refuse an off-grid hour.

OUTPUT (MET_DIR/)
  u.npy, v.npy, omega.npy, T.npy : (nframe, nlev_band, 192, 288) float32,
                                   CESM level order (top -> bottom)
  T_full.npy                     : (nframe, nkrad, 192, 288) float32, the
                                   emulator temperature on EVERY CESM native
                                   level the radiation column uses (p >= 1.005
                                   Pa, radiation.py's krad), at those levels'
                                   hybrid pressures evaluated with the
                                   emulator's own surface pressure -- what the
                                   archive's T on a hybrid level is. For the
                                   CESM-free radiation column (BC_SRC=clim).
  ps.npy                         : (nframe, 192, 288) float32 Pa, for the record
  meta.json                      : hour0, step_hours, nframe, klevs, plev_pa,
                                   lat, lon, provenance (frames dir, ckpt, ...)

USAGE
  python3 scripts/met/emu_to_met.py --frames runs/<tag>/frames_1997-01-01 \
      --t0 1997-01-01T06 --out runs/<tag>/met [--p-lo 1 --p-hi 150]

The band is selected exactly as coupling.py does (P_LO_HPA/P_HI_HPA on the
CESM reference pressures), so the level set matches a run launched with the
same P_LO_HPA/P_HI_HPA.  ps_norm.npy holds the emulator's normalized surface
pressure channel: physical ps = ps_norm * 1e5 Pa (sai_run.py --massfix).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
import _paths  # noqa: E402  (REPO_ROOT, sys.path for coupling's modules)

# The archive the coupled model is built on; only its grid and level
# definitions are read here (one open of the U file), never its fields.
CESM_DIR = os.environ.get('CESM_DIR',
    '/data/cesm2.1.5_output/histSST/'
    'f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001/'
    'archive/atm/proc/tseries')
CESM_PREFIX = os.environ.get('CESM_PREFIX',
    'f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001.cam')
CESM_SUF = os.environ.get('CESM_SUF', '.1996010100-2014123100.nc')
ARCHIVE_T0 = np.datetime64('1996-01-01T00:00:00')   # hour index 0 of the h1 axis
PS_REF = 1.0e5
EMU_COEF = os.environ.get('EMU_COEF', '/data/paradis_jcm/jcm_analytic_p20bt.npz')


def cesm_grid(p_lo_hpa, p_hi_hpa):
    import xarray as xr
    fn = f'{CESM_DIR}/hour_1/{CESM_PREFIX}.h1.U{CESM_SUF}'
    ds = xr.open_dataset(fn)
    lat = ds['lat'].values.astype(np.float64)
    lon = ds['lon'].values.astype(np.float64)
    plev = ds['hyam'].values * ds['P0'].values + ds['hybm'].values * PS_REF
    band = np.where((plev >= p_lo_hpa * 100) & (plev <= p_hi_hpa * 100))[0]
    klevs = list(range(band.min(), band.max() + 1))
    hybm_band = ds['hybm'].values[klevs]
    # the radiation column: every native level with p_ref >= 1.005 Pa (radiation.py)
    krad = np.where(plev >= 1.005)[0]
    hy = dict(hyam=ds['hyam'].values.astype(np.float64), hybm=ds['hybm'].values.astype(np.float64),
              P0=float(ds['P0'].values))
    ds.close()
    assert np.all(hybm_band == 0.0), \
        'band levels are not pure-pressure; the fixed-pressure target is wrong'
    return lat, lon, klevs, plev[klevs], krad, hy


def emu_grid():
    z = np.load(EMU_COEF)
    lat = z['lat'].astype(np.float64)                 # ascending, no poles
    lon = z['lon'].astype(np.float64)
    hyam = z['hyam_pa'].astype(np.float64)            # surface-first
    hybm = z['hybm'].astype(np.float64)
    assert lat[0] < lat[-1] and str(z['LEVEL_ORDER']) == 'surface_first'
    return lat, lon, hyam, hybm


def hours_since_archive_t0(t0):
    t = np.datetime64(t0)
    h = (t - ARCHIVE_T0) / np.timedelta64(1, 'h')
    # the archive is noleap; the emulator's t0 is a real-calendar timestamp.
    # Between 1996-01-01 and the run start, every Feb 29 the real calendar
    # passed is an hour-day the noleap axis does not have.
    leaps = sum(1 for y in range(1996, int(str(t)[:4]) + 1)
                if ((y % 4 == 0 and y % 100 != 0) or y % 400 == 0)
                and np.datetime64(f'{y}-02-29') < t)
    return int(round(h)) - 24 * leaps


def bilinear_weights(src, dst, periodic=False, period=360.0):
    """Index pairs + weights mapping a 1-D ascending source axis onto dst.
    Outside the source range: nearest source node (constant extrapolation)."""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    if periodic:
        ext = np.concatenate([src, [src[0] + period]])
        d = np.mod(dst - src[0], period) + src[0]
        i0 = np.clip(np.searchsorted(ext, d, side='right') - 1, 0, src.size - 1)
        i1 = (i0 + 1) % src.size
        w1 = (d - ext[i0]) / (ext[i0 + 1] - ext[i0])
        return i0, i1, 1.0 - w1, w1
    i0 = np.clip(np.searchsorted(src, dst, side='right') - 1, 0, src.size - 2)
    i1 = i0 + 1
    w1 = np.clip((dst - src[i0]) / (src[i1] - src[i0]), 0.0, 1.0)
    return i0, i1, 1.0 - w1, w1


def regrid_h(f, lat_w, lon_w):
    """f (..., nlat_s, nlon_s) -> (..., nlat_d, nlon_d), bilinear."""
    a0, a1, wa0, wa1 = lat_w
    b0, b1, wb0, wb1 = lon_w
    g = f[..., a0, :] * wa0[:, None] + f[..., a1, :] * wa1[:, None]
    return g[..., :, b0] * wb0 + g[..., :, b1] * wb1


def interp_logp_columns(f, p_src, p_dst, clamp=False):
    """f (nsrc, ncol), p_src (nsrc, ncol) increasing along axis 0, p_dst
    (ndst,) or (ndst, ncol) -> (ndst, ncol), linear in log p per column.
    Targets outside a column's source range are refused (clamp=False) or
    take the end value (clamp=True; the full-column T below the emulator's
    lowest level over high terrain, and its top level)."""
    lp = np.log(p_src); lt = np.log(np.asarray(p_dst, np.float64))
    if lt.ndim == 1:
        lt = np.broadcast_to(lt[:, None], (lt.size, f.shape[1]))
    if not clamp:
        assert np.all(lt[0] >= lp[0]) and np.all(lt[-1] <= lp[-1]), \
            'target pressures outside the emulator column range'
    out = np.empty((lt.shape[0], f.shape[1]), f.dtype)
    cols = np.arange(f.shape[1])
    for j in range(lt.shape[0]):
        t = lt[j]
        # per-column bracketing index: number of source nodes <= t, minus one
        k0 = np.clip((lp <= t[None, :]).sum(axis=0) - 1, 0, lp.shape[0] - 2)
        k1 = k0 + 1
        l0 = lp[k0, cols]; l1 = lp[k1, cols]
        w = (t - l0) / (l1 - l0)
        if clamp:
            w = np.clip(w, 0.0, 1.0)
        out[j] = f[k0, cols] * (1 - w) + f[k1, cols] * w
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--frames', required=True, help='dir with u/v/T/omega/ps_norm .npy')
    ap.add_argument('--t0', required=True, help='time of frame 0, e.g. 1997-01-01T06')
    ap.add_argument('--out', required=True, help='MET_DIR to create')
    ap.add_argument('--step-hours', type=int, default=6)
    ap.add_argument('--p-lo', type=float, default=float(os.environ.get('P_LO_HPA', '1.0')))
    ap.add_argument('--p-hi', type=float, default=float(os.environ.get('P_HI_HPA', '150.0')))
    ap.add_argument('--nframe', type=int, default=0, help='0 = all')
    ap.add_argument('--provenance', default='', help='free text stored in meta.json')
    a = ap.parse_args()

    lat_c, lon_c, klevs, plev_c, krad, hy = cesm_grid(a.p_lo, a.p_hi)
    lat_e, lon_e, hyam, hybm = emu_grid()
    nlev_b = len(klevs)
    nkrad = krad.size

    src = {k: np.load(os.path.join(a.frames, f'{k}.npy'), mmap_mode='r')
           for k in ('u', 'v', 'T', 'omega')}
    ps_norm = np.load(os.path.join(a.frames, 'ps_norm.npy'), mmap_mode='r')
    nt, nz, ny, nx = src['u'].shape
    assert (nz, ny, nx) == (hyam.size, lat_e.size, lon_e.size), src['u'].shape
    if a.nframe:
        nt = min(nt, a.nframe)

    # emulator levels that bracket the band, surface-first indices
    p_ref = hyam + hybm * PS_REF
    kk = np.where((p_ref >= plev_c[0] * 0.5) & (p_ref <= plev_c[-1] * 1.5))[0]
    kk = np.arange(max(kk.min() - 1, 0), min(kk.max() + 2, nz))
    # ascending pressure (= descending index on a surface-first axis)
    ksel = kk[::-1]

    lat_w = bilinear_weights(lat_e, lat_c)
    lon_w = bilinear_weights(lon_e, lon_c, periodic=True)
    npole = int((np.abs(lat_c) > np.abs(lat_e).max()).sum())

    os.makedirs(a.out, exist_ok=True)
    out = {k: np.lib.format.open_memmap(os.path.join(a.out, f'{k}.npy'), mode='w+',
                                        dtype=np.float32, shape=(nt, nlev_b, lat_c.size, lon_c.size))
           for k in ('u', 'v', 'omega', 'T')}
    ps_out = np.lib.format.open_memmap(os.path.join(a.out, 'ps.npy'), mode='w+',
                                       dtype=np.float32, shape=(nt, lat_c.size, lon_c.size))
    tf_out = np.lib.format.open_memmap(os.path.join(a.out, 'T_full.npy'), mode='w+',
                                       dtype=np.float32, shape=(nt, nkrad, lat_c.size, lon_c.size))
    ksel_all = np.arange(nz)[::-1]                                     # whole column, ascending p
    print(f'frames {nt} x emulator L{nz} {ny}x{nx} -> band {nlev_b} levels '
          f'({plev_c[0]/100:.2f}..{plev_c[-1]/100:.2f} hPa) on {lat_c.size}x{lon_c.size}; '
          f'emulator levels used {ksel.min()}..{ksel.max()}; {npole} pole rows clamped', flush=True)
    ncol = ny * nx
    t_start = time.time()
    for i in range(nt):
        ps = np.asarray(ps_norm[i], np.float64) * 1e5                     # Pa
        p_src = (hyam[ksel][:, None] + hybm[ksel][:, None] * ps.reshape(1, ncol))   # (nk, ncol)
        for k in ('u', 'v', 'T', 'omega'):
            f = np.asarray(src[k][i][ksel], np.float64).reshape(ksel.size, ncol)
            g = interp_logp_columns(f, p_src, plev_c).reshape(nlev_b, ny, nx)
            out[k][i] = regrid_h(g, lat_w, lon_w).astype(np.float32)
        ps_out[i] = regrid_h(ps, lat_w, lon_w).astype(np.float32)
        # the full radiation column: emulator T on the CESM hybrid levels at the
        # emulator's ps, ends clamped (surface over high terrain; the top level)
        p_all = hyam[ksel_all][:, None] + hybm[ksel_all][:, None] * ps.reshape(1, ncol)
        p_tgt = hy['hyam'][krad][:, None] * hy['P0'] + hy['hybm'][krad][:, None] * ps.reshape(1, ncol)
        f = np.asarray(src['T'][i][ksel_all], np.float64).reshape(nz, ncol)
        g = interp_logp_columns(f, p_all, p_tgt, clamp=True).reshape(nkrad, ny, nx)
        tf_out[i] = regrid_h(g, lat_w, lon_w).astype(np.float32)
        if i % 200 == 0 or i == nt - 1:
            print(f'  frame {i}/{nt}  {time.time()-t_start:.0f}s', flush=True)
    for m in list(out.values()) + [ps_out, tf_out]:
        m.flush()
    hour0 = hours_since_archive_t0(a.t0)
    meta = dict(source='jcm-emulator', frames=os.path.abspath(a.frames), t0=a.t0,
                hour0=hour0, archive_t0=str(ARCHIVE_T0), calendar='noleap hour index on the CESM h1 axis',
                step_hours=a.step_hours, nframe=int(nt), klevs=[int(k) for k in klevs],
                plev_pa=[float(p) for p in plev_c], p_lo_hpa=a.p_lo, p_hi_hpa=a.p_hi,
                lat=[float(x) for x in lat_c], lon=[float(x) for x in lon_c],
                emu_levels_used=[int(ksel.min()), int(ksel.max())], emu_coef=EMU_COEF,
                krad=[int(k) for k in krad],
                pole_rows_clamped=npole, provenance=a.provenance,
                fields=dict(u='m/s', v='m/s', omega='Pa/s', T='K', ps='Pa', T_full='K on krad'),
                level_order='CESM top->bottom (klevs ascending)')
    with open(os.path.join(a.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'wrote {a.out}: hour0={hour0} (frame 0 = {a.t0}), {nt} frames, {time.time()-t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
