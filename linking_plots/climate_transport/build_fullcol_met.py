#!/usr/bin/env python3
"""Full-column (1.24 hPa .. 1000 hPa) fixed-pressure met for the surface-reset age of air (2026-11-09).

emu_to_met.py only builds pure-pressure CESM band levels (it asserts hybm == 0, i.e. p <= ~80 hPa), but the advection
operator needs FIXED pressure levels, so this builds them directly:
  levels  = the 24 band levels of runs/emu_winds_P2s200_1997_1998/met (1.24..142.99 hPa, identical numbers)
            + 21 fixed levels 160..1000 hPa
  grid    = the band met's 192 x 288 lat/lon (bilinear, the same weights as emu_to_met.regrid_h)
  vertical= linear in log p per emulator column, from its 95 hybrid levels at its own ps (jcm_analytic_p20bt hyam/hybm)
  below the local surface (target p > ps): u = v = omega = 0 (no flow through terrain); ps.npy is kept so the age
            tracer can be reset in every cell at or below the surface.
Frames: global index g = 0..2920 (1997-01-01T06 + 6 h g); g < 1460 from the 1997 frames, g >= 1460 from the 1998 frames
(1998 frame 0 == 1997 frame 1460, checked by concat_met.py).
usage: build_fullcol_met.py OUT [NW]"""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '/data/AIDE-SAI-link/scripts/met')
from emu_to_met import bilinear_weights, regrid_h, interp_logp_columns
OUT = sys.argv[1]; NW = int(sys.argv[2]) if len(sys.argv) > 2 else 6
BAND = '/data/AIDE-SAI-link/runs/emu_winds_P2s200_1997_1998/met'
FR = ['/data/AIDE-SAI-link/runs/emu_winds_P2s200_1997/frames_1997-01-01', '/data/AIDE-SAI-link/runs/emu_winds_P2s200_1998/frames_1998-01-01']
mb = json.load(open(f'{BAND}/meta.json'))
lat_c, lon_c = np.array(mb['lat']), np.array(mb['lon'])
PLEV = np.concatenate([np.array(mb['plev_pa']), np.array([160, 180, 200, 225, 250, 275, 300, 350, 400, 450, 500,
                                                          550, 600, 650, 700, 750, 800, 850, 900, 950, 1000.]) * 100])
assert np.all(np.diff(PLEV) > 0)
CO = np.load('/data/paradis_jcm/jcm_analytic_p20bt.npz')
HYAM, HYBM = CO['hyam_pa'].astype(np.float64), CO['hybm'].astype(np.float64)       # surface-first
lat_e, lon_e = CO['lat'].astype(np.float64), CO['lon'].astype(np.float64)
LW = bilinear_weights(lat_e, lat_c); OW = bilinear_weights(lon_e, lon_c, periodic=True)
NT = 2921; NZ = PLEV.size; NY, NX = lat_c.size, lon_c.size


def src(g):
    return (FR[0], g) if g < 1460 else (FR[1], g - 1460)


def work(g):
    f, i = src(g)
    a = {k: np.load(f'{f}/{k}.npy', mmap_mode='r') for k in ('u', 'v', 'omega', 'ps_norm')}
    ps = np.asarray(a['ps_norm'][i], np.float64) * 1e5                     # (96,192) Pa
    ncol = ps.size
    p_src = (HYAM[::-1][:, None] + HYBM[::-1][:, None] * ps.reshape(1, ncol))   # ascending p
    res = {}
    for k in ('u', 'v', 'omega'):
        x = np.asarray(a[k][i], np.float64)[::-1].reshape(95, ncol)
        y = interp_logp_columns(x, p_src, PLEV, clamp=True)                 # (NZ, ncol); top level clamps above 0.01 hPa never happens
        y[PLEV[:, None] > ps.reshape(1, ncol)] = 0.0                          # below the surface: no flow
        res[k] = regrid_h(y.reshape(NZ, 96, 192), LW, OW).astype(np.float32)
    res['ps'] = regrid_h(ps, LW, OW).astype(np.float32)
    o = {k: np.load(f'{OUT}/{k}.npy', mmap_mode='r+') for k in res}
    for k in res:
        o[k][g] = res[k]
    for m in o.values():
        m.flush()
    return g


if __name__ == '__main__':
    os.nice(15); os.makedirs(OUT, exist_ok=True); t0 = time.time()
    for k in ('u', 'v', 'omega'):
        np.lib.format.open_memmap(f'{OUT}/{k}.npy', mode='w+', dtype=np.float32, shape=(NT, NZ, NY, NX)).flush()
    np.lib.format.open_memmap(f'{OUT}/ps.npy', mode='w+', dtype=np.float32, shape=(NT, NY, NX)).flush()
    frames = range(NT) if not os.environ.get('TEST') else [0, 1460, 2920]
    with Pool(NW) as pool:
        for n, g in enumerate(pool.imap_unordered(work, frames, chunksize=4)):
            if n % 200 == 0: print(f'frame {n}/{len(frames)}  {time.time() - t0:.0f}s', flush=True)
    meta = dict(mb); meta.update(nframe=NT, plev_pa=[float(p) for p in PLEV], p_hi_hpa=1000.0, klevs=None, source='jcm-emulator full column',
                provenance='build_fullcol_met.py: band levels + 21 fixed levels 160-1000 hPa; u=v=omega=0 below the local surface',
                frames=FR, fields=dict(u='m/s', v='m/s', omega='Pa/s', ps='Pa'))
    json.dump(meta, open(f'{OUT}/meta.json', 'w'), indent=1)
    print('wrote', OUT, f'{time.time() - t0:.0f}s')
