"""Per-day reduction of the 4 runs (emulator / numerical, injection / control). One task per day (4 frames).
Writes cache/day_DDD.npz. Read-only on the run directories."""
import os, sys, time
import numpy as np
from multiprocessing import Pool
from common import *

THR_SW = 20.0     # W/m2: frames with toa_sw_dn below this are excluded from the albedo-change ratio
K50 = 17          # band level nearest 50 hPa (51.7 hPa)
_c = {}
SEL = {k: PAIRS[k] for k in os.environ.get('PAIRS_SEL', 'emu,num').split(',')}
K0 = list(SEL)[-1]
PFX = os.environ.get('PFX', '')

def init():
    c = np.load(CLIM)
    m = meta()
    _c['RH'] = c['RELHUM'][:, m['klevs']].astype(np.float32)
    _c['doy'] = c['month_centre_doy']
    kr = m['krad']
    _c['hyam'] = c['hyam'][kr]; _c['hybm'] = c['hybm'][kr]; _c['P0'] = float(c['P0'])
    _c['T'] = np.load(f'{MET}/T.npy', mmap_mode='r')
    _c['Tf'] = np.load(f'{MET}/T_full.npy', mmap_mode='r')
    _c['ps'] = np.load(f'{MET}/ps.npy', mmap_mode='r')

def size_stats(num, mas, w, dpb, mmid, wt):
    """column stats for one frame. num,mas (40,24,ny,nx); w (24,ny,nx) = dp*strat/g."""
    Nk = np.einsum('klyx,lyx->kyx', num, w, dtype=np.float64)          # #/m2
    Mk = np.einsum('klyx,lyx->kyx', mas, w, dtype=np.float64)          # kg/m2
    Ntot = Nk.sum(0); Mtot = Mk.sum(0)
    Dk = np.where(Nk > 0, np.cbrt(np.maximum(Mk, 0) / np.maximum(Nk, 1e-300) / RHO_AER * 6 / np.pi) * 1e9,
                  dpb[:, None, None])                                  # nm, column-mean particle mass per bin
    lnD = np.log(Dk)
    lndg = (Nk * lnD).sum(0) / np.maximum(Ntot, 1e-300)
    sig = np.exp(np.sqrt(np.maximum((Nk * (lnD - lndg) ** 2).sum(0) / np.maximum(Ntot, 1e-300), 0)))
    dmass = np.exp((Mk * lnD).sum(0) / np.maximum(Mtot, 1e-300))
    d = dpb[:, None, None]
    deff_dry = (Nk * d**3).sum(0) / np.maximum((Nk * d**2).sum(0), 1e-300)
    d3 = wet_d3(mmid[:, None, None, None], wt[None]).astype(np.float32)   # m^3
    d2 = d3 ** (2.0 / 3.0)
    nw = num * w[None].astype(np.float32)
    deff_wet = (nw * d3).sum((0, 1), dtype=np.float64) / np.maximum((nw * d2).sum((0, 1), dtype=np.float64), 1e-300) * 1e9
    # column N per bin is also useful for a mean size distribution
    return dict(dg=np.exp(lndg), sig=sig, dmass=dmass, deff_dry=deff_dry, deff_wet=deff_wet), Nk, Mk

def do_day(d):
    out = f'{CACHE}/day_{PFX}{d:03d}.npz'
    if os.path.exists(out):
        return d, 0.0
    t0 = time.time()
    acc = {}
    def add(k, v):
        v = np.asarray(v, np.float64)
        acc[k] = acc.get(k, 0) + v
    for s in range(4 * d, 4 * d + 4):
        # tropopause at the frame's own time (met index s+1); wet growth at the run's T3d time (index s)
        Tf = np.asarray(_c['Tf'][s + 1]); ps = np.asarray(_c['ps'][s + 1])
        pf = _c['hyam'][:, None, None] * _c['P0'] + _c['hybm'][:, None, None] * ps[None]
        nk = Tf.shape[0]
        pt = wmo_tropopause(Tf.reshape(nk, -1), pf.reshape(nk, -1)).reshape(ps.shape)
        nofound = np.isnan(pt)
        ptu = np.where(nofound, np.inf, pt)
        Tb = np.asarray(_c['T'][s]).astype(np.float64)
        m0, m1, wgt = month_weights(H0 + 6 * s, _c['doy'])
        rh = np.clip(((1 - wgt) * _c['RH'][m0] + wgt * _c['RH'][m1]) / 100.0, 0, 1)
        wt = h2so4_equilibrium_wt(Tb, rh * 100.0)
        fr = {}
        for key, (inj, ctl) in SEL.items():
            for role, run in (('inj', inj), ('ctl', ctl)):
                z = np.load(frame_file(run, s))
                assert int(z['step']) == s
                plev = z['plev_pa']; dp = np.gradient(plev)
                strat = plev[:, None, None] < ptu[None]
                w = (dp[:, None, None] * strat) / G
                wall = np.broadcast_to(dp[:, None, None] / G, strat.shape)
                f = {k: z[k] for k in ('toa_sw_up', 'toa_olr', 'toa_sw_dn', 'aod550', 'hr_kday', 'hr_sw_kday', 'sfc_sw_dn', 'sfc_lw_dn')}
                mt = z['mtot']; nt = z['ntot']
                f['colM'] = (mt * w).sum(0) * 1e6               # mg/m2, strat only
                f['colN'] = (nt * w).sum(0)                     # #/m2
                f['colM_all'] = (mt * wall).sum(0) * 1e6
                f['colSO2'] = (z['so2'] * w).sum(0) * 1e6
                f['strat'] = strat
                if role == 'inj':
                    dpb = z['dp_bin_nm']; mmid = (dpb * 1e-9) ** 3 * np.pi / 6 * RHO_AER
                    num = z['num']; mas = z['mas']
                    if num.shape[0] != len(dpb):          # MAM4: modes -> 40 TOMAS bins (mam4_adapter.modes_to_bins)
                        num, mas = mam_modes_to_bins(num, mas, dpb)
                    st, Nk, Mk = size_stats(num, mas, w, dpb, mmid, wt)
                    for k2, v in st.items():
                        add(f'{key}_{k2}', v)
                    add(f'{key}_Nk_gm', (Nk * np.cos(np.deg2rad(z['lat']))[None, :, None]).sum((1, 2)))
                    add(f'{key}_Mk_gm', (Mk * np.cos(np.deg2rad(z['lat']))[None, :, None]).sum((1, 2)))
                    # deff_wet_nm-based column estimate (number-weighted mean of the per-cell values) for comparison
                    nw = nt * w
                    add(f'{key}_deffwet_cellavg', (z['deff_wet_nm'] * nw).sum(0) / np.maximum(nw.sum(0), 1e-300))
                    del num, mas
                fr[(key, role)] = f
                lat = z['lat']
            fi, fc = fr[(key, 'inj')], fr[(key, 'ctl')]
            for k2 in ('toa_sw_up', 'toa_olr', 'aod550', 'colM', 'colN', 'colM_all', 'colSO2', 'sfc_sw_dn', 'sfc_lw_dn'):
                add(f'{key}_inj_{k2}', fi[k2]); add(f'{key}_ctl_{k2}', fc[k2])
            dsw = fi['toa_sw_up'].astype(np.float64) - fc['toa_sw_up']
            day_ = fi['toa_sw_dn'] > THR_SW
            add(f'{key}_dswup_lit', np.where(day_, dsw, 0)); add(f'{key}_swdn_lit', np.where(day_, fi['toa_sw_dn'], 0))
            add(f'{key}_swdn_diff_absmax', 0 * dsw)  # placeholder keeps keys aligned
            acc[f'{key}_swdn_diff_absmax'] = np.maximum(acc.get(f'{key}_swdn_diff_absmax_v', 0),
                                                         np.abs(fi['toa_sw_dn'] - fc['toa_sw_dn']).max())
            acc[f'{key}_swdn_diff_absmax_v'] = acc[f'{key}_swdn_diff_absmax']
            m = fi['strat']
            for hk in ('hr_kday', 'hr_sw_kday'):
                dh = fi[hk].astype(np.float64) - fc[hk]
                add(f'{key}_d{hk}_zsum', (dh * m).sum(2)); 
                add(f'{key}_d{hk}_50', np.where(m[K50], dh[K50], np.nan))
                add(f'{key}_d{hk}_colint', (dh * m * (np.gradient(plev)[:, None, None])).sum(0) / G)  # K/day * kg/m2
            add(f'{key}_hr_inj_zsum', (fi['hr_kday'] * m).sum(2))
            add(f'{key}_hr_ctl_zsum', (fc['hr_kday'] * m).sum(2))
        add('strat_zcount', fr[(K0, 'inj')]['strat'].sum(2))
        add('strat_frac_cells', fr[(K0, 'inj')]['strat'].mean(0))
        add('ptrop', np.where(nofound, np.nan, pt))
        add('nofound', nofound)
        add('swdn', fr[(K0, 'inj')]['toa_sw_dn'])
        add('tband_wt', wt)  # wt% for reference
    res = {k: (v / 4.0).astype(np.float32) for k, v in acc.items()
           if not k.endswith('_v') and not k.endswith('absmax')}
    res.update({k: np.float32(v) for k, v in acc.items() if k.endswith('absmax')})
    res['day'] = np.int32(d)
    np.savez(out + '.tmp.npz', **res)
    os.replace(out + '.tmp.npz', out)
    return d, time.time() - t0

if __name__ == '__main__':
    days = range(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else range(NDAYS)
    nw = int(os.environ.get('NW', '24'))
    t0 = time.time()
    with Pool(nw, initializer=init) as p:
        for d, dt in p.imap_unordered(do_day, list(days)):
            print(f'day {d} {dt:.0f}s  elapsed {time.time() - t0:.0f}s', flush=True)
