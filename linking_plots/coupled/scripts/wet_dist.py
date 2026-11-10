"""Annual global-mean stratospheric size distribution vs WET diameter, same Tabazadeh/Tang prescription for all schemes.
One frame per day (s = 4d+1), all 6 runs. Per stratospheric cell and bin: Dw from settling.wet_size on the bin mass MMID
with wt% = Tabazadeh(T = emulator band T at met index s, RH = bc_clim RELHUM month-interpolated at hour H0+6s), as in
coupling.py's frame diagnostics. Weights n * dp/g * strat * cos(lat) / sum(cos) / nlon -> global-mean per m2."""
import os, sys, time
import numpy as np
from multiprocessing import Pool
from common import *

EDGES = np.logspace(np.log10(1.0), np.log10(30000.0), 81)        # nm, 80 bins
DLN = np.log(EDGES[1] / EDGES[0])
WT0, DWT, NWT = 10.0, 0.25, 281      # wt% classes 10..80 in 0.25 steps
_c = {}
RUNSET = [('emu', 'inj', 'r1_tomas_emu'), ('emu', 'ctl', 'r1c_tomas_emu_ctl'), ('num', 'inj', 'r2_tomas_fast'),
          ('num', 'ctl', 'r2c_tomas_fast_ctl'), ('mam', 'inj', 'r3_mam4'), ('mam', 'ctl', 'r3c_mam4_ctl')]

def init():
    c = np.load(CLIM); m = meta()
    _c['RH'] = c['RELHUM'][:, m['klevs']].astype(np.float32); _c['doy'] = c['month_centre_doy']
    kr = m['krad']; _c['hyam'] = c['hyam'][kr]; _c['hybm'] = c['hybm'][kr]; _c['P0'] = float(c['P0'])
    for k in ('T', 'T_full', 'ps'):
        _c[k] = np.load(f'{MET}/{k}.npy', mmap_mode='r')

def hist(D, wts):
    idx = np.clip(np.searchsorted(EDGES, D.ravel()) - 1, 0, 79)
    return np.bincount(idx, weights=wts.ravel(), minlength=80)[:80]

def do_day(d):
    out = f'{CACHE}/wetdist/v2_day_{d:03d}.npz'
    if os.path.exists(out):
        return d, 0
    t0 = time.time(); s = 4 * d + 1
    Tf = np.asarray(_c['T_full'][s + 1]); ps = np.asarray(_c['ps'][s + 1])
    pf = _c['hyam'][:, None, None] * _c['P0'] + _c['hybm'][:, None, None] * ps[None]
    pt = wmo_tropopause(Tf.reshape(Tf.shape[0], -1), pf.reshape(Tf.shape[0], -1)).reshape(ps.shape)
    ptu = np.where(np.isnan(pt), np.inf, pt)
    Tb = np.asarray(_c['T'][s]).astype(np.float64)
    m0, m1, wg = month_weights(H0 + 6 * s, _c['doy'])
    rh = np.clip(((1 - wg) * _c['RH'][m0] + wg * _c['RH'][m1]) / 100.0, 0, 1)
    wt = h2so4_equilibrium_wt(Tb, rh * 100.0)
    res = {}
    for key, role, run in RUNSET:
        z = np.load(frame_file(run, s))
        plev = z['plev_pa']; dp = np.gradient(plev); lat = z['lat']; dpb = z['dp_bin_nm']
        mmid = (dpb * 1e-9) ** 3 * np.pi / 6 * RHO_AER
        strat = plev[:, None, None] < ptu[None]
        cw = np.cos(np.deg2rad(lat)); cw = cw / (cw.sum() * len(z['lon']))
        w = (dp[:, None, None] * strat / G) * cw[None, :, None]           # (lev,lat,lon): per-m2 global-mean weight
        num = z['num']; mas = z['mas']
        if num.shape[0] != len(dpb):
            num, mas = mam_modes_to_bins(num, mas, dpb)
        nw = num.astype(np.float64) * w[None]                              # (40,lev,lat,lon) #/m2
        mw = mas.astype(np.float64) * w[None]                              # kg SO4/m2
        d3 = wet_d3(mmid[:, None, None, None], wt[None])                   # m^3
        Dw = np.cbrt(d3) * 1e9                                             # nm
        # wet particle mass (SO4 + water) per particle, for the wet-mass-weighted diameter
        acid = mmid[:, None, None, None] * (98.0 / 96.0); mwet = mmid[:, None, None, None] + acid * (100.0 / wt[None] - 1.0)
        p = f'{key}_{role}_'
        # number per (TOMAS bin, wt% class): the wet-size spectrum is rebuilt from it in plot_wet_dist.py by spreading
        # each bin uniformly in lnD over its own WET edge interval (edges xk_k, xk_k+1 grown at the class wt%)
        cls = np.clip(np.rint((wt - WT0) / DWT).astype(int), 0, NWT - 1).ravel()
        res[p + 'Nkc'] = np.stack([np.bincount(cls, weights=nw[k].ravel(), minlength=NWT) for k in range(nw.shape[0])])
        Dd = np.broadcast_to(dpb[:, None, None, None], Dw.shape)
        gf = Dw / Dd
        res[p + 'sums'] = np.array([nw.sum(), (nw * Dw**3).sum(), (nw * Dw**2).sum(),
                                    (nw * mwet).sum(), (nw * mwet * np.log(Dw)).sum(),
                                    (nw * gf).sum(), mw.sum(), (mw * gf).sum(),
                                    (nw * Dd**3).sum(), (nw * Dd**2).sum()])
    ww = (np.gradient(plev)[:, None, None] * strat) * cw[None, :, None]
    res['TRH'] = np.array([(Tb * ww).sum() / ww.sum(), (rh * ww).sum() / ww.sum(), (wt * ww).sum() / ww.sum(), ww.sum()])
    np.savez(out, day=d, **res)
    return d, time.time() - t0

if __name__ == '__main__':
    with Pool(int(os.environ.get('NW', '36')), initializer=init) as p:
        for d, dt in p.imap_unordered(do_day, range(NDAYS)):
            print(d, f'{dt:.0f}s', flush=True)
