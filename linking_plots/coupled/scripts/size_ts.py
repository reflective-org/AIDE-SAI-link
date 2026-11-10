"""Daily stratospheric column size distribution (per bin N and M), area-weighted over global / NH / SH, for the
injection runs (also the 15-45N and 15-45S bands) of the emulator, numerical and MAM4 schemes (2-year coupled campaign). One frame per day (step 4d+1,
the same stratosphere filter as process.py: WMO lapse-rate tropopause from the emulator T at the frame's own time).
MAM4 modes are mapped onto the 40 TOMAS bins with mam_modes_to_bins, as in process.py.
Writes cache/size_ts/day_DDD.npz; aggregate with --merge -> cache/size_ts_v2.npz.  Read-only on the run dirs.
usage: LP_NDAYS=730 LP_MET=.../emu_winds_P2s200_1997_1998/met python size_ts.py [d0 d1]    |    python size_ts.py --merge"""
import os, sys, time, glob
import numpy as np
from multiprocessing import Pool
from common import *

OUTD = f'{CACHE}/size_ts'
KEYS = ('emu', 'num', 'mam')
_c = {}


def init():
    c = np.load(CLIM); m = meta(); kr = m['krad']
    _c['hyam'] = c['hyam'][kr]; _c['hybm'] = c['hybm'][kr]; _c['P0'] = float(c['P0'])
    _c['Tf'] = np.load(f'{MET}/T_full.npy', mmap_mode='r'); _c['ps'] = np.load(f'{MET}/ps.npy', mmap_mode='r')


def do_day(d):
    out = f'{OUTD}/day_{d:03d}.npz'
    if os.path.exists(out):
        return d, 0.0
    t0 = time.time(); s = 4 * d + 1
    Tf = np.asarray(_c['Tf'][s + 1]); ps = np.asarray(_c['ps'][s + 1])
    pf = _c['hyam'][:, None, None] * _c['P0'] + _c['hybm'][:, None, None] * ps[None]
    nk = Tf.shape[0]
    pt = wmo_tropopause(Tf.reshape(nk, -1), pf.reshape(nk, -1)).reshape(ps.shape)
    ptu = np.where(np.isnan(pt), np.inf, pt)
    res = {}
    for key in KEYS:
        z = np.load(frame_file(PAIRS[key][0], s)); assert int(z['step']) == s
        plev = z['plev_pa']; dpb = z['dp_bin_nm']; lat = z['lat']
        w = (np.gradient(plev)[:, None, None] * (plev[:, None, None] < ptu[None])) / G       # kg m-2 per level, strat only
        num = z['num']; mas = z['mas']
        if num.shape[0] != len(dpb):
            num, mas = mam_modes_to_bins(num, mas, dpb)
        Nk = np.einsum('klyx,lyx->kyx', num, w, dtype=np.float64)                             # # m-2
        Mk = np.einsum('klyx,lyx->kyx', mas, w, dtype=np.float64)                             # kg m-2
        cw = np.cos(np.deg2rad(lat))[:, None] * np.ones(Nk.shape[2])[None]
        for reg, msk in (('glob', np.ones_like(lat, bool)), ('nh', lat > 0), ('sh', lat < 0),
                         ('n1545', (lat >= 15) & (lat <= 45)), ('s1545', (lat >= -45) & (lat <= -15))):
            ww = cw * msk[:, None]; ww = ww / ww.sum()
            res[f'{key}_N_{reg}'] = (Nk * ww).sum((1, 2)); res[f'{key}_M_{reg}'] = (Mk * ww).sum((1, 2))
        del num, mas
    res['dp_bin_nm'] = dpb; res['day'] = d; res['step'] = s
    os.makedirs(OUTD, exist_ok=True)
    np.savez(out + '.tmp.npz', **res); os.replace(out + '.tmp.npz', out)
    return d, time.time() - t0


if __name__ == '__main__':
    if sys.argv[1:2] == ['--merge']:
        fs = sorted(glob.glob(f'{OUTD}/day_[0-9][0-9][0-9].npz')); zs = [np.load(f) for f in fs]
        M = {k: np.stack([z[k] for z in zs]) for k in zs[0].files if k not in ('dp_bin_nm',)}
        M['dp_bin_nm'] = zs[0]['dp_bin_nm']
        np.savez(f'{CACHE}/size_ts_v2.npz', **M); print('merged', len(fs), 'days ->', f'{CACHE}/size_ts_v2.npz'); sys.exit()
    days = range(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else range(NDAYS)
    t0 = time.time()
    with Pool(int(os.environ.get('NW', '24')), initializer=init) as p:
        for d, dt in p.imap_unordered(do_day, list(days)):
            print(f'day {d} {dt:.0f}s  elapsed {time.time() - t0:.0f}s', flush=True)
